"""
korean_query_generation.py — Automatically add Korean queries to queries.jsonl.
Generated from entity templates. No hand-crafting required.
Run once, then delete this file.
"""
import sys as _sys; _sys.exit(print(__doc__ or "")) if any(_a in ("-h", "--help") for _a in _sys.argv[1:]) else None   # usage

import os, pathlib; REPO_ROOT = pathlib.Path(os.environ.get("FRESHCACHE_ROOT", ".")).resolve(); import json, uuid; import os, pathlib; DATA_ROOT = pathlib.Path(os.environ.get("DATA_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "data")).resolve(); OUTPUT_ROOT = pathlib.Path(os.environ.get("OUTPUT_DIR") or os.path.join(os.environ.get("FRESHCACHE_ROOT", "."), "outputs")).resolve()
from pathlib import Path
from collections import Counter

QUERIES_FILE = ((DATA_ROOT / "queries.jsonl"))

def record(query, fc):
    return {
        "query_id":           f"q_{uuid.uuid4().hex[:8]}",
        "query":              query.strip(),
        "freshness_class":    fc,
        "language":           "ko",
        "source":             "ko_template",
        "topic":              "",
        "original_freshness": "",
    }

# (template, entities) — {e} is replaced by each entity.
# If entities is empty list, template is used as-is.
TEMPLATES = {

    "REAL_TIME": [
        ("현재 {e} 주가는 얼마입니까?",
         ["삼성전자","카카오","네이버","현대자동차","LG전자","SK하이닉스",
          "셀트리온","포스코홀딩스","기아","롯데케미칼","삼성SDI",
          "LG화학","현대모비스","KB금융","신한지주"]),
        ("지금 {e} 날씨는 어떻습니까?",
         ["서울","부산","대구","인천","광주","대전","울산","제주","수원","창원"]),
        ("달러 대비 {e} 환율은 현재 얼마입니까?",
         ["원화","유로","엔화","위안화","파운드"]),
        ("지금 {e} 대기질 지수는 얼마입니까?",
         ["서울","부산","인천","대구","광주"]),
        ("오늘 {e} 경기 실시간 스코어는 어떻게 됩니까?",
         ["KBO","K리그","NBA","프리미어리그"]),
        ("현재 {e} 가격은 얼마입니까?",
         ["비트코인","이더리움","금","은","국제유가"]),
        ("지금 코스피 지수는 얼마입니까?",             []),
        ("현재 코스닥 지수는 얼마입니까?",             []),
        ("오늘 원달러 환율은 얼마입니까?",             []),
        ("현재 서울 실시간 대기질 지수는 얼마입니까?", []),
    ],

    "FAST": [
        ("{e} 최신 앨범은 무엇입니까?",
         ["BTS","블랙핑크","아이브","뉴진스","에스파","세븐틴",
          "스트레이키즈","트와이스","NCT","엑소","르세라핌","케플러","(여자)아이들"]),
        ("{e} 최근 경기 결과는 어떻습니까?",
         ["삼성 라이온즈","LG 트윈스","두산 베어스",
          "KIA 타이거즈","한화 이글스","SSG 랜더스"]),
        ("현재 {e} 드라마 시청률 1위는 무엇입니까?",
         ["tvN","MBC","SBS","KBS2","JTBC"]),
        ("{e} 오늘의 주요 뉴스는 무엇입니까?",
         ["한국","경제","스포츠","연예","정치"]),
        ("현재 {e} 음원 차트 1위는 무엇입니까?",
         ["멜론","지니","벅스","플로"]),
        ("이번 주 한국 박스오피스 1위 영화는 무엇입니까?",          []),
        ("이번 시즌 K리그1 순위는 어떻게 됩니까?",                  []),
        ("이번 시즌 KBO 리그 순위는 어떻게 됩니까?",                []),
        ("현재 유튜브 한국 인기 급상승 동영상 1위는 무엇입니까?",   []),
        ("오늘 한국 박스오피스 순위는 어떻게 됩니까?",              []),
    ],

    "MEDIUM": [
        ("한국의 최근 {e} 정책 동향은 어떻습니까?",
         ["인공지능","반도체","전기차","재생에너지","탄소중립",
          "바이오헬스","핀테크","우주항공"]),
        ("최신 {e} 기술 트렌드는 무엇입니까?",
         ["대형언어모델","자율주행","양자컴퓨팅","로봇공학","AR·VR"]),
        ("한국의 최근 {e} 현황은 어떻습니까?",
         ["물가상승률","실업률","수출입 통계","부동산 시장","기준금리"]),
        ("올해 한국 {e} 전망은 어떻습니까?",
         ["경제성장률","수출","고용시장","부동산"]),
        ("최근 발표된 한국 {e} 관련 정책은 무엇입니까?",
         ["개인정보","AI 규제","플랫폼","가상자산"]),
        ("이번 시즌 {e} 최신 트렌드는 무엇입니까?",
         ["K뷰티","K푸드","한국 패션","한국 여행지"]),
        ("올해 누리호 발사 현황은 어떻습니까?",     []),
        ("최근 한국 반도체 수출 현황은?",            []),
        ("현재 한국 기준금리는 얼마입니까?",         []),
        ("올해 한국 경제성장률 전망은 얼마입니까?",  []),
    ],

    "SLOW": [
        ("현행 한국 {e}의 주요 내용은 무엇입니까?",
         ["개인정보 보호법","저작권법","근로기준법","환경법",
          "의료법","교육기본법","소비자보호법"]),
        ("한국의 현재 {e} 기준은 어떻게 됩니까?",
         ["최저임금","건강보험료율","국민연금 보험료율","법정 근로시간"]),
        ("한국 {e} 관련 현황 통계는 어떻습니까?",
         ["인구","출생률","고령화율","대학 진학률","도시화율"]),
        ("현재 한국의 {e}는 몇 개입니까?",
         ["국회의원 의석 수","광역시 수","국립대학교 수","상장 기업 수"]),
        ("{e} 관련 한국 현행 제도는 무엇입니까?",
         ["병역의무","국민건강보험","국민연금","기초생활보장"]),
        ("현재 한국의 최저임금은 얼마입니까?",               []),
        ("현재 한국 인구는 몇 명입니까?",                    []),
        ("한국의 현행 법정 근로시간은 몇 시간입니까?",       []),
        ("현재 한국 국민연금 보험료율은 몇 퍼센트입니까?",   []),
    ],

    "TIMELESS": [
        ("{e}의 원리는 무엇입니까?",
         ["광합성","열역학 제2법칙","진화론","특수상대성이론",
          "양자역학","중력의 법칙","전자기 유도","핵분열","삼투압"]),
        ("{e}란 무엇입니까?",
         ["피타고라스 정리","뉴턴의 운동법칙","DNA 구조",
          "원소 주기율표","미적분학","확률론","집합론","세포의 구조"]),
        ("{e}의 역사적 의의는 무엇입니까?",
         ["훈민정음 창제","3·1운동","대한민국 건국",
          "한국전쟁","1988 서울올림픽","새마을운동"]),
        ("한국의 {e}는 어디에 위치합니까?",
         ["백두산","한라산","독도","울릉도","DMZ"]),
        ("{e}의 화학식은 무엇입니까?",
         ["물","이산화탄소","포도당","암모니아","황산"]),
        ("{e}의 정의는 무엇입니까?",
         ["삼권분립","민주주의","시장경제","법치주의","인권"]),
    ],
}


def main():
    existing = []
    if QUERIES_FILE.exists():
        with open(QUERIES_FILE, encoding="utf-8") as f:
            existing = [json.loads(l) for l in f]

    seen = {q["query"].strip().lower() for q in existing}
    print(f"Existing queries : {len(existing)}")

    new_records = []
    added_per_class = Counter()

    for fc, template_list in TEMPLATES.items():
        for template, entities in template_list:
            if not entities:
                q = template.strip()
                if q.lower() not in seen:
                    seen.add(q.lower())
                    new_records.append(record(q, fc))
                    added_per_class[fc] += 1
            else:
                for entity in entities:
                    q = template.format(e=entity).strip()
                    if q.lower() not in seen:
                        seen.add(q.lower())
                        new_records.append(record(q, fc))
                        added_per_class[fc] += 1

    all_records = existing + new_records
    with open(QUERIES_FILE, "w", encoding="utf-8") as f:
        for r in all_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Korean queries added : {len(new_records)}")
    print(f"Added per class      : {dict(added_per_class)}")
    print(f"\nFinal dataset:")
    print(f"  Total    : {len(all_records)}")
    print(f"  Per class: {dict(Counter(r['freshness_class'] for r in all_records))}")
    print(f"  Language : {dict(Counter(r['language'] for r in all_records))}")


if __name__ == "__main__":
    main()