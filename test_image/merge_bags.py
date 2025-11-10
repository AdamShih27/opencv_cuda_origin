#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
merge_bags.py
將多個被 --split 切分的 rosbag 依「訊息寫入時間 t」做全域排序後合併成單一 bag。

需求：
- ROS 1 (Noetic) 的 Python3 環境
- 套件：rosbag, roslib, rospy

使用：
  python3 merge_bags.py -o out.bag in1.bag in2.bag ...
  python3 merge_bags.py -o out.bag path/to/*.bag
"""

import argparse
import glob
import heapq
import os
import sys
from collections import defaultdict

try:
    import rosbag
except ImportError:
    print("[ERROR] 找不到 rosbag 模組。請在 ROS Noetic (Python3) 環境下執行。")
    sys.exit(1)

import rospy  # ← 重要：使用 rospy.Time 來建立時間戳


def resolve_input_files(inputs):
    """展開萬用字元並去重、檢查存在性。"""
    files = []
    for pat in inputs:
        matched = glob.glob(pat)
        if not matched and os.path.isfile(pat):
            matched = [pat]
        files.extend(matched)
    # 去重並排序（先用檔名排序，稍後再以 bag 開始時間排序）
    files = sorted(set(files))
    if not files:
        raise FileNotFoundError("沒有找到任何輸入 bag。")
    return files


def sort_by_bag_time(files):
    """依 bag 的 get_start_time() 排序；若失敗，回退為檔名排序。"""
    decorated = []
    for f in files:
        try:
            with rosbag.Bag(f, 'r', allow_unindexed=True) as b:
                st = b.get_start_time()
        except Exception:
            # 若讀不到時間，給個很大的值，避免影響其他正常檔案排序
            st = float('inf')
        decorated.append((st, f))
    decorated.sort(key=lambda x: (x[0], x[1]))
    return [f for _, f in decorated]


def open_bag_reader(path, topics=None, exclude=None):
    """
    開啟一個 bag 讀取器，回傳 (bag_obj, iterator)。
    topics: 只包含的主題清單
    exclude: 要排除的主題清單（在 read_messages 之後過濾）
    """
    bag = rosbag.Bag(path, 'r', allow_unindexed=True)
    # 先用 topics 參數粗篩，再在迭代時用 exclude 精篩
    it = bag.read_messages(topics=topics)
    if exclude:
        def _gen():
            for topic, msg, t in it:
                if topic in exclude:
                    continue
                yield topic, msg, t
        return bag, _gen()
    else:
        return bag, it


def compression_arg_to_rosbag(compression_str):
    """字串轉 rosbag.Compression。"""
    if not compression_str or compression_str.lower() == 'none':
        return rosbag.Compression.NONE
    cs = compression_str.lower()
    if cs == 'bz2':
        return rosbag.Compression.BZ2
    if cs == 'lz4':
        # 需 ROS/系統支援 lz4
        try:
            return rosbag.Compression.LZ4
        except AttributeError:
            print("[WARN] 目前 rosbag 不支援 LZ4，改用 NONE。")
            return rosbag.Compression.NONE
    print(f"[WARN] 未知壓縮方式：{compression_str}，改用 NONE。")
    return rosbag.Compression.NONE


def human_time_range(bag_path):
    """回傳 (start, end) 秒（float），讀不到則回傳 (None, None)。"""
    try:
        with rosbag.Bag(bag_path, 'r', allow_unindexed=True) as b:
            return b.get_start_time(), b.get_end_time()
    except Exception:
        return None, None


def merge_bags(output_path, input_files, topics_keep=None, topics_exclude=None,
               compression='none', dry_run=False, verbose=True):
    """
    以全域時間順序合併多個 bag。
    使用最小堆（heapq）從每個 bag 的迭代器彈出最早的訊息寫入輸出。
    """
    comp_mode = compression_arg_to_rosbag(compression)
    readers = []
    heap = []
    counts = defaultdict(int)
    per_topic_counts = defaultdict(int)

    # 開啟所有輸入 bag 讀取器
    for idx, f in enumerate(input_files):
        try:
            b, it = open_bag_reader(f, topics=topics_keep, exclude=topics_exclude)
            readers.append((f, b, it))
        except Exception as e:
            print(f"[WARN] 無法開啟 {f}: {e}")
            continue

    if not readers:
        raise RuntimeError("沒有任何可讀取的 bag。")

    # 預先從每個 reader 取出一筆放入 heap
    # heap 元素：(t.to_sec(), seq_id, reader_index, topic, msg)
    # 加入 seq_id 防止 t 相同時的比較問題
    seq = 0
    for i, (f, b, it) in enumerate(readers):
        try:
            topic, msg, t = next(it)
            heapq.heappush(heap, (t.to_sec(), seq, i, topic, msg))
            seq += 1
        except StopIteration:
            # 空 bag
            pass
        except Exception as e:
            print(f"[WARN] 讀取 {f} 時發生錯誤：{e}")

    if not heap:
        raise RuntimeError("輸入 bag 皆無訊息。")

    if dry_run:
        print("[DRY-RUN] 不輸出檔案，只做掃描與統計。")
        out_bag = None
    else:
        out_bag = rosbag.Bag(output_path, 'w', compression=comp_mode)

    prev_t = None
    total = 0

    try:
        while heap:
            t_sec, _, ridx, topic, msg = heapq.heappop(heap)

            # 寫入（改用 rospy.Time）
            if out_bag is not None:
                out_bag.write(topic, msg, rospy.Time.from_sec(t_sec))

            total += 1
            counts[ridx] += 1
            per_topic_counts[topic] += 1

            # 監控非遞增時間（理論上允許，但提示使用者）
            if prev_t is not None and t_sec < prev_t:
                # 只提示一次以免洗版
                if verbose:
                    print(f"[WARN] 時間非遞增：{t_sec:.6f} < {prev_t:.6f}（資料可能重疊或時鐘回退）")
                    verbose = False  # 之後不再重複提示
            prev_t = t_sec

            # 從該 reader 再取下一筆
            f, b, it = readers[ridx]
            try:
                topic, msg, t = next(it)
                seq += 1
                heapq.heappush(heap, (t.to_sec(), seq, ridx, topic, msg))
            except StopIteration:
                pass
            except Exception as e:
                print(f"[WARN] 讀取 {f} 下一筆時出錯：{e}")

    finally:
        # 關閉所有 bag
        for _, b, _ in readers:
            try:
                b.close()
            except Exception:
                pass
        if out_bag is not None:
            try:
                out_bag.close()
            except Exception:
                pass

    # 報告
    print("\n=== 合併完成 ===")
    print(f"輸出檔案：{output_path if output_path else '(dry-run)'}")
    print(f"總訊息數：{total:,}")
    print("各輸入檔案訊息數：")
    for i, (f, _, _) in enumerate(readers):
        st, et = human_time_range(f)
        if st is not None and et is not None:
            print(f"  - [{i}] {os.path.basename(f)} : {counts[i]:,} msgs | "
                  f"range={st:.3f}~{et:.3f} ({et-st:.3f}s)")
        else:
            print(f"  - [{i}] {os.path.basename(f)} : {counts[i]:,} msgs | range=unknown")

    print("各 topic 訊息數：")
    for tp, c in sorted(per_topic_counts.items(), key=lambda x: (-x[1], x[0])):
        print(f"  - {tp} : {c:,}")


def main():
    parser = argparse.ArgumentParser(
        description="將多個被 --split 切分的 rosbag 依時間順序合併成一個檔案。")
    parser.add_argument("inputs", nargs="+",
                        help="輸入 bag（可多個或使用萬用字元，例如 *.bag）")
    parser.add_argument("-o", "--output", required=True,
                        help="輸出 bag 檔名，例如 merged.bag")
    parser.add_argument("--topics", nargs="*", default=None,
                        help="只保留這些 topics（白名單）。不指定則保留全部。")
    parser.add_argument("--exclude", nargs="*", default=None,
                        help="排除這些 topics（黑名單）。")
    parser.add_argument("--compression", choices=["none", "bz2", "lz4"],
                        default="none", help="輸出壓縮方式（預設 none）")
    parser.add_argument("--dry-run", action="store_true",
                        help="僅掃描與統計，不真的寫出檔案。")
    args = parser.parse_args()

    try:
        files = resolve_input_files(args.inputs)
        files = sort_by_bag_time(files)
        print("將依下列順序處理（由 bag 起始時間排序）：")
        for i, f in enumerate(files):
            st, et = human_time_range(f)
            if st is not None and et is not None:
                rng = f"range={st:.3f}~{et:.3f} ({et-st:.3f}s)"
            else:
                rng = "range=unknown"
            print(f"  [{i}] {f}  {rng}")

        merge_bags(
            output_path=args.output,
            input_files=files,
            topics_keep=args.topics,
            topics_exclude=args.exclude,
            compression=args.compression,
            dry_run=args.dry_run,
        )
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(2)


if __name__ == "__main__":
    main()
