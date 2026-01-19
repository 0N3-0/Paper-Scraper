import re
import json
import os
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any

import arxiv

try:
    from openai import OpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

CATEGORIES = {
    "AI": ["cs.AI", "cs.LG", "cs.CV", "cs.CL", "cs.NE"],
    "Security": ["cs.CR", "cs.IT", "math.IT"],
}

DOWNLOADED_PAPERS_FILE = "downloaded_papers.json"

AI_CLIENT = None
if OPENAI_AVAILABLE:
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        AI_CLIENT = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )


def load_downloaded_papers() -> Dict[str, int]:
    if os.path.exists(DOWNLOADED_PAPERS_FILE):
        try:
            with open(DOWNLOADED_PAPERS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_downloaded_papers(downloaded: Dict[str, int]) -> None:
    with open(DOWNLOADED_PAPERS_FILE, "w", encoding="utf-8") as f:
        json.dump(downloaded, f, ensure_ascii=False, indent=2)


def is_new_version(
    pdf_url: str, current_version: int, downloaded: Dict[str, int]
) -> bool:
    if pdf_url not in downloaded:
        return True
    return current_version > downloaded[pdf_url]


def get_version_from_pdf_url(pdf_url: str | None) -> int:
    if not pdf_url:
        return 1
    match = re.search(r"v(\d+)", pdf_url)
    if match:
        return int(match.group(1))
    return 1


def ai_summarize_zh(text: str) -> str:
    if not AI_CLIENT or not text:
        return ""
    try:
        response = AI_CLIENT.chat.completions.create(
            model="qwen-max",
            messages=[
                {
                    "role": "system",
                    "content": "用3-4句简洁的中文概括以下学术论文摘要（200字以内），抓住：1）研究问题和目标；2）主要方法或技术；3）核心贡献或创新点；4）关键结果。直接明了;5) 避免翻译腔，且不要引入原文未出现的信息，无需背景介绍。",
                },
                {"role": "user", "content": text[:4000]},
            ],
            max_tokens=300,
            temperature=0.3,
        )
        content = response.choices[0].message.content
        return content.strip() if content else ""
    except Exception:
        return ""


def search_papers(category: str, max_results: int = 50) -> List[arxiv.Result]:
    category_codes = CATEGORIES[category]
    queries = [f"cat:{cat}" for cat in category_codes]
    query = " OR ".join(queries)

    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.LastUpdatedDate,
    )

    client = arxiv.Client()
    results = list(client.results(search))
    return results


def get_paper_info(paper: arxiv.Result) -> Dict[str, Any]:
    return {
        "title": paper.title,
        "authors": [author.name for author in paper.authors],
        "summary": paper.summary,
        "published": paper.published,
        "updated": paper.updated,
        "version": get_version_from_pdf_url(paper.pdf_url),
        "pdf_url": paper.pdf_url,
        "doi": paper.doi,
        "primary_category": paper.primary_category,
        "comment": paper.comment,
        "journal_ref": paper.journal_ref,
    }


def filter_recent_by_published(
    papers: List[arxiv.Result], days: int = 180
) -> List[arxiv.Result]:
    date_limit = datetime.now(timezone.utc) - timedelta(days=days)
    recent_papers = []
    for paper in papers:
        if paper.published.replace(tzinfo=timezone.utc) >= date_limit:
            recent_papers.append(paper)
    return recent_papers


def filter_recent_by_updated(
    papers: List[arxiv.Result], days: int = 180, min_version: int = 2
) -> List[arxiv.Result]:
    date_limit = datetime.now(timezone.utc) - timedelta(days=days)
    recent_papers = []
    for paper in papers:
        updated_date = paper.updated.replace(tzinfo=timezone.utc)
        version = get_version_from_pdf_url(paper.pdf_url)
        if updated_date >= date_limit and version >= min_version:
            recent_papers.append(paper)
    return recent_papers


def format_papers_for_email(
    updated_papers: List[Dict[str, Any]],
    published_papers: List[Dict[str, Any]],
) -> str:
    lines = []
    lines.append(f"日期: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"新论文: {len(updated_papers) + len(published_papers)} 篇\n")
    lines.append("=" * 34)

    count = 1
    for paper in updated_papers:
        lines.append(f"\n【更新】论文 #{count}")
        lines.append(f"标题: {paper['title']}")
        lines.append(f"作者: {', '.join(paper['authors'][:5])}")
        if len(paper["authors"]) > 5:
            lines.append(f"      ... 等 {len(paper['authors'])} 位作者")
        lines.append(f"发布时间: {paper['published'].strftime('%Y-%m-%d')}")
        lines.append(f"更新日期: {paper['updated'].strftime('%Y-%m-%d')}")
        lines.append(f"版本: v{paper['version']}")
        lines.append(f"分类: {paper['primary_category']}")
        if paper.get("doi"):
            lines.append(f"DOI: {paper['doi']}")
        if paper.get("comment"):
            lines.append(f"注释: {paper['comment']}")
        if paper.get("journal_ref"):
            lines.append(f"期刊引用: {paper['journal_ref']}")
        summary = ai_summarize_zh(paper["summary"])
        if summary:
            lines.append(f"\n要点: {summary}")
        lines.append(f"\n完整摘要 [{len(paper['summary'])} 字符]:")
        lines.append(paper["summary"][:1000])
        if len(paper["summary"]) > 1000:
            lines.append(f"... (共 {len(paper['summary'])} 字符)")
        lines.append(f"\n链接: {paper['pdf_url']}")
        lines.append("=" * 34)
        count += 1

    for paper in published_papers:
        lines.append(f"\n【发布】论文 #{count}")
        lines.append(f"标题: {paper['title']}")
        lines.append(f"作者: {', '.join(paper['authors'][:5])}")
        if len(paper["authors"]) > 5:
            lines.append(f"      ... 等 {len(paper['authors'])} 位作者")
        lines.append(f"发布时间: {paper['published'].strftime('%Y-%m-%d')}")
        lines.append(f"版本: v{paper['version']}")
        lines.append(f"分类: {paper['primary_category']}")
        if paper.get("doi"):
            lines.append(f"DOI: {paper['doi']}")
        if paper.get("comment"):
            lines.append(f"注释: {paper['comment']}")
        if paper.get("journal_ref"):
            lines.append(f"期刊引用: {paper['journal_ref']}")
        summary = ai_summarize_zh(paper["summary"])
        if summary:
            lines.append(f"\n要点: {summary}")
        lines.append(f"\n完整摘要 [{len(paper['summary'])} 字符]:")
        lines.append(paper["summary"][:1000])
        if len(paper["summary"]) > 1000:
            lines.append(f"... (共 {len(paper['summary'])} 字符)")
        lines.append(f"\n链接: {paper['pdf_url']}")
        lines.append("=" * 34)
        count += 1

    return "\n".join(lines)


def send_email_via_qq(
    subject: str,
    body: str,
    recipient: str,
) -> bool:
    sender = os.environ.get("QQ_EMAIL")
    auth_code = os.environ.get("QQ_EMAIL_AUTH_CODE")

    if not sender or not auth_code:
        print("警告: 未设置 QQ_EMAIL 或 QQ_EMAIL_AUTH_CODE 环境变量")
        return False

    smtp_server = "smtp.qq.com"
    smtp_port = 587

    try:
        msg = MIMEMultipart()
        msg["From"] = sender
        msg["To"] = recipient
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain", "utf-8"))

        context = ssl.create_default_context()
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls(context=context)
            server.login(sender, auth_code)
            server.sendmail(sender, recipient, msg.as_string())

        print(f"邮件已发送到 {recipient}")
        return True
    except Exception as e:
        print(f"邮件发送失败: {e}")
        return False


def main():
    downloaded = load_downloaded_papers()

    candidate_updated: List[Dict[str, Any]] = []
    candidate_published: List[Dict[str, Any]] = []

    for category in ["AI", "Security"]:
        try:
            papers = search_papers(category, max_results=50)

            recent_published = filter_recent_by_published(papers, days=180)
            recent_updated = filter_recent_by_updated(papers, days=180, min_version=2)

            updated_titles = set()
            for paper in recent_updated:
                info = get_paper_info(paper)
                if not is_new_version(info["pdf_url"], info["version"], downloaded):
                    continue
                candidate_updated.append(info)
                updated_titles.add(info["title"])

            for paper in recent_published:
                info = get_paper_info(paper)
                if info["title"] not in updated_titles:
                    if not is_new_version(info["pdf_url"], info["version"], downloaded):
                        continue
                    candidate_published.append(info)
        except Exception:
            continue

    candidate_updated.sort(key=lambda x: x["updated"], reverse=True)
    candidate_published.sort(key=lambda x: x["published"], reverse=True)

    all_papers_by_updated = candidate_updated[:2]
    all_papers_by_published = candidate_published[:2]

    for info in all_papers_by_updated + all_papers_by_published:
        downloaded[info["pdf_url"]] = info["version"]

    save_downloaded_papers(downloaded)

    recipient = os.environ.get("RECIPIENT_EMAIL")
    if recipient:
        email_body = format_papers_for_email(
            all_papers_by_updated, all_papers_by_published
        )
        send_email_via_qq(
            f"学术论文速递 - {datetime.now().strftime('%Y-%m-%d')}",
            email_body,
            recipient,
        )


if __name__ == "__main__":
    main()
