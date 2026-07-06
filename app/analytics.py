from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path


@dataclass
class AnalyticsSummary:
    total_turns: int
    memory_hits: int
    memory_misses: int
    hit_rate: float
    avg_top_similarity: float
    topic_counts: dict[str, int]
    top_queries: list[str]


class AnalyticsService:
    def __init__(self, log_path: Path = Path("logs/turns.jsonl")) -> None:
        self.log_path = log_path

    def _load_records(self) -> list[dict]:
        if not self.log_path.exists():
            return []

        records: list[dict] = []
        for line in self.log_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return records

    def summarize(self) -> AnalyticsSummary:
        records = self._load_records()
        total = len(records)
        if total == 0:
            return AnalyticsSummary(0, 0, 0, 0.0, 0.0, {}, [])

        hits = sum(1 for r in records if r.get("memory_hit") is True)
        misses = total - hits
        avg_similarity = sum(float(r.get("top_similarity", 0.0)) for r in records) / total

        topic_counter = Counter(r.get("topic", "general") for r in records)
        query_counter = Counter(r.get("query", "").strip() for r in records if r.get("query", "").strip())

        return AnalyticsSummary(
            total_turns=total,
            memory_hits=hits,
            memory_misses=misses,
            hit_rate=hits / total,
            avg_top_similarity=avg_similarity,
            topic_counts=dict(topic_counter.most_common()),
            top_queries=[q for q, _ in query_counter.most_common(10)],
        )

    def write_dashboard_files(self, out_dir: Path = Path("logs")) -> tuple[Path, Path]:
        out_dir.mkdir(parents=True, exist_ok=True)
        summary = self.summarize()

        summary_json_path = out_dir / "analytics_summary.json"
        dashboard_html_path = out_dir / "dashboard.html"

        summary_json_path.write_text(
            json.dumps(
                {
                    "total_turns": summary.total_turns,
                    "memory_hits": summary.memory_hits,
                    "memory_misses": summary.memory_misses,
                    "hit_rate": round(summary.hit_rate, 4),
                    "avg_top_similarity": round(summary.avg_top_similarity, 4),
                    "topic_counts": summary.topic_counts,
                    "top_queries": summary.top_queries,
                },
                ensure_ascii=True,
                indent=2,
            ),
            encoding="utf-8",
        )

        topic_rows = "".join(
            f"<tr><td>{topic}</td><td>{count}</td></tr>" for topic, count in summary.topic_counts.items()
        )
        query_rows = "".join(f"<li>{q}</li>" for q in summary.top_queries)

        html = f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Memory Agent Analytics Dashboard</title>
  <style>
    :root {{
      --bg: #f8f7f4;
      --card: #ffffff;
      --ink: #1b1e24;
      --muted: #6a7280;
      --accent: #0b8f77;
      --accent-2: #c2882f;
      --line: #e5e7eb;
    }}
    body {{
      margin: 0;
      font-family: Georgia, Cambria, \"Times New Roman\", serif;
      color: var(--ink);
      background: radial-gradient(circle at top right, #e8f4ef, var(--bg) 40%);
    }}
    .wrap {{ max-width: 980px; margin: 30px auto; padding: 0 16px 32px; }}
    .hero {{ margin-bottom: 16px; }}
    .hero h1 {{ margin: 0; font-size: 2rem; }}
    .hero p {{ margin: 6px 0 0; color: var(--muted); }}
    .grid {{ display: grid; gap: 12px; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); margin-bottom: 16px; }}
    .card {{ background: var(--card); border: 1px solid var(--line); border-radius: 14px; padding: 14px; }}
    .metric {{ font-size: 1.7rem; font-weight: bold; }}
    .label {{ color: var(--muted); font-size: 0.9rem; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ border-bottom: 1px solid var(--line); text-align: left; padding: 8px 6px; }}
    th {{ color: var(--muted); font-weight: 600; }}
    ul {{ margin-top: 8px; }}
    .tag {{ display: inline-block; padding: 2px 8px; border-radius: 999px; background: #eef7f4; color: var(--accent); }}
    .miss {{ background: #fff5e9; color: var(--accent-2); }}
  </style>
</head>
<body>
  <div class=\"wrap\">
    <section class=\"hero\">
      <h1>Memory-First Agent Dashboard</h1>
      <p>Turn analytics, retrieval quality, and topic distribution.</p>
    </section>

    <section class=\"grid\">
      <article class=\"card\"><div class=\"metric\">{summary.total_turns}</div><div class=\"label\">Total turns</div></article>
      <article class=\"card\"><div class=\"metric\">{summary.memory_hits}</div><div class=\"label\">Memory hits <span class=\"tag\">hit</span></div></article>
      <article class=\"card\"><div class=\"metric\">{summary.memory_misses}</div><div class=\"label\">Memory misses <span class=\"tag miss\">miss</span></div></article>
      <article class=\"card\"><div class=\"metric\">{summary.hit_rate * 100:.1f}%</div><div class=\"label\">Hit rate</div></article>
      <article class=\"card\"><div class=\"metric\">{summary.avg_top_similarity:.3f}</div><div class=\"label\">Avg top similarity</div></article>
    </section>

    <section class=\"card\" style=\"margin-bottom:12px;\">
      <h2>Topics</h2>
      <table>
        <thead><tr><th>Topic</th><th>Count</th></tr></thead>
        <tbody>{topic_rows}</tbody>
      </table>
    </section>

    <section class=\"card\">
      <h2>Top Questions</h2>
      <ul>{query_rows}</ul>
    </section>
  </div>
</body>
</html>
"""

        dashboard_html_path.write_text(html, encoding="utf-8")
        return summary_json_path, dashboard_html_path
