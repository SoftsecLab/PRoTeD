from __future__ import annotations

import time
import urllib.error
import urllib.request
from pathlib import Path



RAW_ROOT = Path("data/raw")


CHEAT_FILES = [
    "ieee-init.jsonl",
    "ieee-chatgpt-generation.jsonl",
    "ieee-chatgpt-polish.jsonl",
    "ieee-chatgpt-fusion.jsonl",
]


DETECTRL_FILES = [
    # Multi-domain
    "multi_domains_arxiv_test.json",
    "multi_domains_arxiv_train.json",
    "multi_domains_writing_prompt_test.json",
    "multi_domains_writing_prompt_train.json",
    "multi_domains_xsum_test.json",
    "multi_domains_xsum_train.json",
    "multi_domains_yelp_review_test.json",
    "multi_domains_yelp_review_train.json",

    # Multi-LLM
    "multi_llms_ChatGPT_test.json",
    "multi_llms_ChatGPT_train.json",
    "multi_llms_Claude-instant_test.json",
    "multi_llms_Claude-instant_train.json",
    "multi_llms_Google-PaLM_test.json",
    "multi_llms_Google-PaLM_train.json",
    "multi_llms_Llama-2-70b_test.json",
    "multi_llms_Llama-2-70b_train.json",
]


def download_file(
    url: str,
    output_path: Path,
    retries: int = 5,
    timeout: int = 120,
) -> bool:

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and output_path.stat().st_size > 0:
        size_mb = output_path.stat().st_size / 1024 / 1024
        print(f"[跳过] 已存在：{output_path} ({size_mb:.2f} MB)")
        return True

    temp_path = output_path.with_suffix(output_path.suffix + ".part")

    if temp_path.exists():
        temp_path.unlink()

    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 PRoTeD-Dataset-Downloader",
            "Accept": "*/*",
        },
    )

    for attempt in range(1, retries + 1):
        try:
            print(
                f"[下载] {output_path.name} "
                f"({attempt}/{retries})"
            )

            with urllib.request.urlopen(
                request,
                timeout=timeout,
            ) as response, temp_path.open("wb") as file:
                while True:
                    chunk = response.read(1024 * 1024)

                    if not chunk:
                        break

                    file.write(chunk)

            if not temp_path.exists() or temp_path.stat().st_size == 0:
                raise RuntimeError("下载后的文件为空")

            temp_path.replace(output_path)

            size_mb = output_path.stat().st_size / 1024 / 1024
            print(
                f"[完成] {output_path} "
                f"({size_mb:.2f} MB)"
            )
            return True

        except (
            urllib.error.URLError,
            urllib.error.HTTPError,
            TimeoutError,
            ConnectionError,
            RuntimeError,
        ) as exc:
            print(f"[失败] {output_path.name}: {exc}")

            if temp_path.exists():
                temp_path.unlink()

            if attempt < retries:
                wait_seconds = attempt * 5
                print(f"       {wait_seconds} 秒后重试")
                time.sleep(wait_seconds)

    print(f"[放弃] 连续 {retries} 次下载失败：{url}")
    return False


def download_cheat() -> list[Path]:

    base_url = (
        "https://raw.githubusercontent.com/"
        "botianzhe/CHEAT/main/data"
    )

    output_dir = RAW_ROOT / "CHEAT"
    failed_files: list[Path] = []

    print("\n========== 下载 CHEAT ==========")

    for filename in CHEAT_FILES:
        output_path = output_dir / filename
        url = f"{base_url}/{filename}"

        if not download_file(url, output_path):
            failed_files.append(output_path)

    return failed_files


def download_detectrl() -> list[Path]:

    base_url = (
        "https://raw.githubusercontent.com/"
        "NLP2CT/DetectRL/main/Benchmark/Tasks/Task1"
    )

    output_dir = (
        RAW_ROOT
        / "DetectRL_original"
    )

    failed_files: list[Path] = []

    print("\n========== 下载 DetectRL ==========")

    for filename in DETECTRL_FILES:
        output_path = output_dir / filename
        url = f"{base_url}/{filename}"

        if not download_file(url, output_path):
            failed_files.append(output_path)

    return failed_files


def print_summary(failed_files: list[Path]) -> None:

    total = len(CHEAT_FILES) + len(DETECTRL_FILES)
    failed = len(failed_files)
    succeeded = total - failed

    print("\n========== 下载结果 ==========")
    print(f"计划下载：{total} 个文件")
    print(f"下载成功：{succeeded} 个文件")
    print(f"下载失败：{failed} 个文件")

    if failed_files:
        print("\n失败文件：")
        for path in failed_files:
            print(f"  - {path}")

        raise SystemExit(1)

    print("\n全部数据文件下载完成。")


def main() -> None:
    failed_files = []

    failed_files.extend(download_cheat())
    failed_files.extend(download_detectrl())

    print_summary(failed_files)


if __name__ == "__main__":
    main()