#!/home/claw/.openclaw/workspace/.venv/bin/python3
import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

import lancedb
import pandas as pd
import requests

def log_error(msg):
    print(f"[ERROR] {msg}", file=sys.stderr)

def log_info(msg):
    print(f"[INFO] {msg}", file=sys.stderr)

def get_embedding(text, url, key, model):
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    
    payload = {
        "model": model,
        "input": text
    }
    
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return data["data"][0]["embedding"]
    except Exception as e:
        log_error(f"Embedding API request failed: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="rag-core: 通用向量数据库管理工具 (Python 版)")
    parser.add_argument("-d", "--db", default="./rag_data", help="数据库根路径")
    parser.add_argument("--api-url", help="Embedding API URL")
    parser.add_argument("--api-key", help="Embedding API Key")
    
    subparsers = parser.add_subparsers(dest="command")
    
    # Init
    subparsers.add_parser("init", help="初始化数据库环境")
    
    # Table
    table_parser = subparsers.add_parser("table", help="表管理操作")
    table_sub = table_parser.add_subparsers(dest="action")
    
    table_sub.add_parser("list", help="列出所有表")
    
    new_parser = table_sub.add_parser("new", help="创建新表")
    new_parser.add_argument("name", help="表名")
    new_parser.add_argument("--dim", type=int, default=1024, help="向量维度")
    new_parser.add_argument("--model", required=True, help="关联模型名")
    
    del_parser = table_sub.add_parser("delete", help="删除表")
    del_parser.add_argument("name", help="表名")
    
    info_parser = table_sub.add_parser("info", help="获取表信息")
    info_parser.add_argument("name", help="表名")
    
    # Data operations
    ingest_parser = subparsers.add_parser("ingest", help="入库数据")
    ingest_parser.add_argument("-t", "--table", required=True, help="目标表名")
    ingest_parser.add_argument("file", help="数据文本文件路径")
    ingest_parser.add_argument("--name", required=True, help="来源标识")
    
    query_parser = subparsers.add_parser("query", help="语义查询")
    query_parser.add_argument("-t", "--table", required=True, help="目标表名")
    query_parser.add_argument("text", help="查询文本")
    query_parser.add_argument("-l", "--limit", type=int, default=5, help="返回数量")
    
    search_parser = subparsers.add_parser("search", help="按 name 精确查找匹配 ID")
    search_parser.add_argument("-t", "--table", required=True, help="目标表名")
    search_parser.add_argument("--name", required=True, help="来源标识")
    
    delete_data_parser = subparsers.add_parser("delete", help="按 ID 删除记录")
    delete_data_parser.add_argument("-t", "--table", required=True, help="目标表名")
    delete_data_parser.add_argument("id", help="记录 ID")
    
    clear_parser = subparsers.add_parser("clear", help="清空表")
    clear_parser.add_argument("-t", "--table", required=True, help="目标表名")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return

    db = lancedb.connect(args.db)
    csv_writer = csv.writer(sys.stdout)

    if args.command == "init":
        os.makedirs(args.db, exist_ok=True)
        log_info(f"数据库目录已初始化: {args.db}")
        
    elif args.command == "table":
        if args.action == "list":
            tables = db.list_tables()
            csv_writer.writerow(["table", "count"])
            for tname in tables:
                tbl = db.open_table(tname)
                # 使用 len(tbl) 或 count_rows() 获取更高效的统计
                csv_writer.writerow([tname, tbl.count_rows()])
                
        elif args.action == "new":
            # 存一份元数据到对应的隐藏文件
            meta = {"dim": args.dim, "model": args.model}
            Path(args.db).mkdir(parents=True, exist_ok=True)
            with open(Path(args.db) / f"{args.name}.meta.json", "w") as f:
                json.dump(meta, f)
            log_info(f"表元数据已保存: {args.name} (将在首次 Ingest 时正式创建)")
            
        elif args.action == "delete":
            db.drop_table(args.name)
            meta_file = Path(args.db) / f"{args.name}.meta.json"
            if meta_file.exists(): meta_file.unlink()
            log_info(f"表已删除: {args.name}")
            
        elif args.action == "info":
            meta_file = Path(args.db) / f"{args.name}.meta.json"
            model, dim = "unknown", "unknown"
            if meta_file.exists():
                with open(meta_file, "r") as f:
                    m = json.load(f)
                    model, dim = m.get("model"), m.get("dim")
            tbl = db.open_table(args.name)
            count = tbl.count_rows()
            csv_writer.writerow(["table", "dimensions", "embedding_model", "item_count"])
            csv_writer.writerow([args.name, dim, model, count])

    elif args.command == "ingest":
        if not args.api_url:
            log_error("--api-url is required for ingest")
            sys.exit(1)
        
        meta_file = Path(args.db) / f"{args.table}.meta.json"
        if not meta_file.exists():
            log_error(f"表 {args.table} 的元数据不存在，请先执行 table new")
            sys.exit(1)
            
        with open(meta_file, "r") as f:
            meta = json.load(f)
        
        with open(args.file, "r", encoding="utf-8") as f:
            text = f.read()
            
        vec = get_embedding(text, args.api_url, args.api_key, meta["model"])
        doc_id = hashlib.sha256((text + args.name).encode()).hexdigest()
        
        data = [{"id": doc_id, "name": args.name, "text": text, "vector": vec}]
        
        if args.table in db.table_names():
            tbl = db.open_table(args.table)
            tbl.add(data)
        else:
            db.create_table(args.table, data)
            
        csv_writer.writerow(["status", "id", "name"])
        csv_writer.writerow(["success", doc_id, args.name])

    elif args.command == "query":
        if not args.api_url:
            log_error("--api-url is required for query")
            sys.exit(1)
            
        meta_file = Path(args.db) / f"{args.table}.meta.json"
        if not meta_file.exists():
            log_error(f"无法获取模型信息，请确保 {args.table}.meta.json 存在")
            sys.exit(1)
            
        with open(meta_file, "r") as f:
            meta = json.load(f)
            
        vec = get_embedding(args.text, args.api_url, args.api_key, meta["model"])
        tbl = db.open_table(args.table)
        
        results = tbl.search(vec).limit(args.limit).to_pandas()
        
        csv_writer.writerow(["score", "id", "name", "text"])
        for _, row in results.iterrows():
            score = 1 - row.get("_distance", 1.0)
            clean_text = str(row["text"]).replace("\n", " ")
            csv_writer.writerow([f"{score:.4f}", row["id"], row["name"], clean_text])

    elif args.command == "search":
        tbl = db.open_table(args.table)
        # 简单过滤搜索
        res = tbl.to_pandas()
        matches = res[res["name"] == args.name]
        csv_writer.writerow(["id", "name"])
        for _, row in matches.iterrows():
            csv_writer.writerow([row["id"], row["name"]])

    elif args.command == "delete":
        tbl = db.open_table(args.table)
        tbl.delete(f'id = "{args.id}"')
        log_info(f"记录已删除: {args.id}")

    elif args.command == "clear":
        tbl = db.open_table(args.table)
        tbl.delete("true")
        log_info(f"表已清空: {args.table}")

if __name__ == "__main__":
    main()
