#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
查看实验日志工具
"""

import os
import glob
import argparse
from datetime import datetime


def list_log_files(logs_dir='results/logs'):
    """列出所有日志文件"""
    if not os.path.exists(logs_dir):
        print(f"❌ 日志目录不存在: {logs_dir}")
        return []
    
    log_files = glob.glob(os.path.join(logs_dir, "*.log"))
    if not log_files:
        print(f"📁 {logs_dir} 目录中没有找到日志文件")
        return []
    
    # 按修改时间排序
    log_files.sort(key=os.path.getmtime, reverse=True)
    
    print(f"📋 找到 {len(log_files)} 个日志文件：")
    print("-" * 80)
    
    for i, log_file in enumerate(log_files, 1):
        basename = os.path.basename(log_file)
        size = os.path.getsize(log_file)
        mtime = datetime.fromtimestamp(os.path.getmtime(log_file))
        
        # 提取分布名称
        distribution = basename.split('_')[0] if '_' in basename else 'unknown'
        
        print(f"{i:2d}. {basename}")
        print(f"    📊 分布: {distribution}")
        print(f"    📅 时间: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"    📦 大小: {size:,} bytes")
        print()
    
    return log_files


def view_log_file(log_file, lines=None):
    """查看日志文件内容"""
    if not os.path.exists(log_file):
        print(f"❌ 日志文件不存在: {log_file}")
        return
    
    print(f"📖 查看日志文件: {os.path.basename(log_file)}")
    print("=" * 80)
    
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            content = f.readlines()
        
        if lines:
            if lines > 0:
                # 显示前N行
                content = content[:lines]
                print(f"📄 显示前 {lines} 行:")
            else:
                # 显示后N行
                content = content[lines:]
                print(f"📄 显示后 {abs(lines)} 行:")
        
        for line in content:
            print(line.rstrip())
            
    except Exception as e:
        print(f"❌ 读取日志文件失败: {e}")


def search_logs(keyword, logs_dir='results/logs'):
    """在日志中搜索关键词"""
    log_files = glob.glob(os.path.join(logs_dir, "*.log"))
    
    print(f"🔍 在 {len(log_files)} 个日志文件中搜索关键词: '{keyword}'")
    print("=" * 80)
    
    found_count = 0
    
    for log_file in log_files:
        basename = os.path.basename(log_file)
        matches = []
        
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    if keyword.lower() in line.lower():
                        matches.append((line_num, line.strip()))
        except Exception as e:
            print(f"❌ 读取 {basename} 失败: {e}")
            continue
        
        if matches:
            print(f"\n📁 {basename}:")
            for line_num, line in matches[:5]:  # 最多显示5个匹配
                print(f"   {line_num:4d}: {line}")
            if len(matches) > 5:
                print(f"   ... 还有 {len(matches) - 5} 个匹配")
            found_count += len(matches)
    
    if found_count == 0:
        print("🚫 没有找到匹配的内容")
    else:
        print(f"\n✅ 共找到 {found_count} 个匹配")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='查看联邦学习实验日志')
    parser.add_argument('--list', '-l', action='store_true', help='列出所有日志文件')
    parser.add_argument('--view', '-v', type=str, help='查看指定的日志文件')
    parser.add_argument('--lines', '-n', type=int, help='显示行数 (正数=前N行, 负数=后N行)')
    parser.add_argument('--search', '-s', type=str, help='搜索关键词')
    parser.add_argument('--latest', action='store_true', help='查看最新的日志文件')
    parser.add_argument('--dir', default='results/logs', help='日志目录 (默认: results/logs)')
    
    args = parser.parse_args()
    
    if args.list or (not args.view and not args.search and not args.latest):
        log_files = list_log_files(args.dir)
        
        if log_files and not args.view and not args.search and not args.latest:
            try:
                choice = input("\n请输入要查看的日志文件编号 (按回车退出): ").strip()
                if choice and choice.isdigit():
                    idx = int(choice) - 1
                    if 0 <= idx < len(log_files):
                        view_log_file(log_files[idx], args.lines)
                    else:
                        print("❌ 无效的编号")
            except KeyboardInterrupt:
                print("\n👋 已退出")
    
    elif args.view:
        # 如果是数字，则按编号查看
        if args.view.isdigit():
            log_files = list_log_files(args.dir)
            idx = int(args.view) - 1
            if 0 <= idx < len(log_files):
                view_log_file(log_files[idx], args.lines)
            else:
                print("❌ 无效的编号")
        else:
            # 否则按文件名查看
            log_file = os.path.join(args.dir, args.view)
            if not log_file.endswith('.log'):
                log_file += '.log'
            view_log_file(log_file, args.lines)
    
    elif args.latest:
        log_files = list_log_files(args.dir)
        if log_files:
            print("📖 查看最新的日志文件...")
            view_log_file(log_files[0], args.lines)
    
    elif args.search:
        search_logs(args.search, args.dir)


if __name__ == '__main__':
    main()
