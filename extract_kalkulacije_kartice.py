import argparse
import csv
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

try:
    import pdfplumber
except Exception:  # pragma: no cover - optional dependency
    pdfplumber = None


DATE_RE = re.compile(r"\b(\d{2}\.\d{2}\.\d{4})\b")
DOC_RE = re.compile(r"^([A-Z]{1,3}-[0-9A-Z-]+)")
NUM_RE = re.compile(r"[-+]?\d{1,3}(?:\.\d{3})*(?:,\d+)?|[-+]?\d+(?:,\d+)?")
SP_MM_RE = re.compile(r"\bSP-MM-\d+\b", re.IGNORECASE)


@dataclass
class StockEvent:
    date: datetime
    qty: float
    doc: str


@dataclass
class KarticaRow:
    sku: str
    article: str
    date: datetime
    doc: str
    opis: str
    referenca: str
    prijem_kolicina: float
    prijem_vrednost: float
    izdavanje_kolicina: float
    izdavanje_vrednost: float
    cena: float
    stanje_kolicina: float
    stanje_vrednost: float

    @property
    def delta_kolicina(self) -> float:
        return self.prijem_kolicina - self.izdavanje_kolicina

    @property
    def doc_tip(self) -> str:
        tip = (self.doc or "").strip().upper()
        if tip.startswith("PS-"):
            return "PS"
        if tip.startswith("IS-"):
            return "IS"
        return ""


def _normalize(text: str) -> str:
    text = str(text).strip()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return text


def _num_from_token(token: str) -> float:
    token = token.replace(".", "").replace(",", ".")
    try:
        return float(token)
    except ValueError:
        return 0.0


def _extract_sku_from_text(text: str) -> str:
    match = re.search(r"\(([A-Za-z0-9-]+)\)", text)
    if match:
        return match.group(1).strip()
    return _normalize(text)


def _parse_date(value) -> datetime | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, datetime):
        return value
    text = str(value).strip().rstrip(".")
    if not text:
        return None
    for fmt in (
        "%d.%m.%Y",
        "%d.%m.%Y %H:%M",
        "%d.%m.%Y %H:%M:%S",
        "%Y-%m-%d",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S.%f",
    ):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def _extract_description_and_reference(line: str, date_match: re.Match) -> tuple[str, str]:
    tail = line[date_match.end() :].strip()
    if not tail:
        return "", ""
    ref_match = SP_MM_RE.search(tail)
    referenca = ref_match.group(0).upper() if ref_match else ""

    first_num = NUM_RE.search(tail)
    opis = tail[: first_num.start()].strip() if first_num else tail.strip()
    return opis, referenca


def _parse_kartica_numeric_tail(line: str) -> tuple[float, float, float, float, float, float, float] | None:
    nums = NUM_RE.findall(line)
    if len(nums) < 7:
        return None
    tail = nums[-7:]
    prijem_kolicina = _num_from_token(tail[0])
    prijem_vrednost = _num_from_token(tail[1])
    izdavanje_kolicina = _num_from_token(tail[2])
    izdavanje_vrednost = _num_from_token(tail[3])
    cena = _num_from_token(tail[4])
    stanje_kolicina = _num_from_token(tail[5])
    stanje_vrednost = _num_from_token(tail[6])
    return (
        prijem_kolicina,
        prijem_vrednost,
        izdavanje_kolicina,
        izdavanje_vrednost,
        cena,
        stanje_kolicina,
        stanje_vrednost,
    )


def parse_kartica_pdf(pdf_path: Path) -> list[KarticaRow]:
    if pdfplumber is None:
        raise RuntimeError("pdfplumber is required for PDF parsing")

    rows: list[KarticaRow] = []
    current_article = ""
    current_sku = ""

    with pdfplumber.open(str(pdf_path)) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            for raw in text.splitlines():
                line = raw.strip()
                if not line:
                    continue

                if line.startswith("ARTIKAL:"):
                    current_article = line.replace("ARTIKAL:", "", 1).strip()
                    current_sku = _extract_sku_from_text(current_article) or current_article
                    continue

                if not current_sku:
                    continue

                # Initial balance line: "Početno stanje na dan 01.04.2024: 0 0,00 0,00"
                norm = _normalize(line).lower()
                if norm.startswith("pocetno stanje na dan"):
                    date_match = DATE_RE.search(line)
                    if not date_match:
                        continue
                    after_colon = line.split(":", 1)[1] if ":" in line else line
                    nums = NUM_RE.findall(after_colon)
                    if not nums:
                        continue
                    stanje_kolicina = _num_from_token(nums[0])
                    stanje_vrednost = _num_from_token(nums[1]) if len(nums) > 1 else 0.0
                    dt = datetime.strptime(date_match.group(1), "%d.%m.%Y")
                    rows.append(
                        KarticaRow(
                            sku=current_sku,
                            article=current_article,
                            date=dt,
                            doc="POCETNO",
                            opis="Početno stanje",
                            referenca="",
                            prijem_kolicina=0.0,
                            prijem_vrednost=0.0,
                            izdavanje_kolicina=0.0,
                            izdavanje_vrednost=0.0,
                            cena=0.0,
                            stanje_kolicina=stanje_kolicina,
                            stanje_vrednost=stanje_vrednost,
                        )
                    )
                    continue

                doc_match = DOC_RE.match(line)
                date_match = DATE_RE.search(line)
                if not doc_match or not date_match:
                    continue

                parsed = _parse_kartica_numeric_tail(line)
                if parsed is None:
                    continue

                opis, referenca = _extract_description_and_reference(line, date_match)
                dt = datetime.strptime(date_match.group(1), "%d.%m.%Y")
                (
                    prijem_kolicina,
                    prijem_vrednost,
                    izdavanje_kolicina,
                    izdavanje_vrednost,
                    cena,
                    stanje_kolicina,
                    stanje_vrednost,
                ) = parsed
                rows.append(
                    KarticaRow(
                        sku=current_sku,
                        article=current_article,
                        date=dt,
                        doc=doc_match.group(1),
                        opis=opis,
                        referenca=referenca,
                        prijem_kolicina=prijem_kolicina,
                        prijem_vrednost=prijem_vrednost,
                        izdavanje_kolicina=izdavanje_kolicina,
                        izdavanje_vrednost=izdavanje_vrednost,
                        cena=cena,
                        stanje_kolicina=stanje_kolicina,
                        stanje_vrednost=stanje_vrednost,
                    )
                )

    rows.sort(key=lambda r: (r.sku, r.date, r.doc))
    return rows


def extract_kalkulacije(
    excel_path: Path, output_dir: Path, sheet_name: str = "Minimax"
) -> tuple[Path, Path]:
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    orig_cols = list(df.columns)
    norm_map = {c: _normalize(c) for c in orig_cols}

    reverse = {}
    for orig, norm in norm_map.items():
        reverse.setdefault(norm, orig)

    def get_col(norm_name: str) -> str | None:
        return reverse.get(norm_name)

    vp_col = get_col("VP")
    if vp_col is None:
        raise ValueError("Missing 'VP' column")

    vp_norm = df[vp_col].astype(str).str.replace(" ", "").str.upper()
    calc = df.loc[vp_norm == "PS"].copy()

    want_norm = [
        "VP",
        "Broj",
        "Datum",
        "Opis",
        "Artikal",
        "Kolicina",
        "Nabavna vrednost",
        "Prodajna vrednost",
        "Prodajna cena",
        "U skladiste",
        "ID prometa u zalihama",
    ]
    missing = [n for n in want_norm if get_col(n) is None]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    cols = {n: get_col(n) for n in want_norm}
    calc = calc[[cols[n] for n in want_norm]].copy()
    calc.columns = want_norm

    calc["Marza"] = calc["Prodajna vrednost"] - calc["Nabavna vrednost"]
    calc["Broj"] = calc["Broj"].astype(str).str.strip()
    calc["Artikal"] = calc["Artikal"].astype(str).str.strip()
    calc["Opis"] = calc["Opis"].astype(str).str.strip()
    calc["U skladiste"] = calc["U skladiste"].astype(str).str.strip()
    calc["SKU"] = calc["Artikal"].apply(_extract_sku_from_text)

    out_detail = output_dir / "kalkulacije_marza.csv"
    calc.to_csv(out_detail, index=False)

    # Aggregation by SKU + article name
    agg = (
        calc.assign(
            Kolicina=calc["Kolicina"].fillna(0),
            Nabavna=calc["Nabavna vrednost"].fillna(0),
            Prodajna=calc["Prodajna vrednost"].fillna(0),
            Marza=calc["Marza"].fillna(0),
        )
        .groupby(["SKU", "Artikal"], dropna=False)
        .agg(
            kolicina_sum=("Kolicina", "sum"),
            nabavna_sum=("Nabavna", "sum"),
            prodajna_sum=("Prodajna", "sum"),
            marza_sum=("Marza", "sum"),
            doc_count=("Broj", "nunique"),
            first_date=("Datum", "min"),
            last_date=("Datum", "max"),
        )
        .reset_index()
    )

    out_agg = output_dir / "kalkulacije_marza_agregat.csv"
    agg.to_csv(out_agg, index=False)
    return out_detail, out_agg


def extract_zero_intervals(pdf_path: Path, output_dir: Path) -> Path:
    parsed = parse_kartica_pdf(pdf_path)
    return write_zero_intervals_from_parsed(parsed, output_dir)


def write_zero_intervals_from_parsed(parsed: list[KarticaRow], output_dir: Path) -> Path:
    events: dict[str, list[StockEvent]] = {}
    article_names: dict[str, str] = {}
    for row in parsed:
        events.setdefault(row.sku, []).append(
            StockEvent(date=row.date, qty=row.stanje_kolicina, doc=row.doc)
        )
        if row.article:
            article_names[row.sku] = row.article

    rows = []
    for sku, items in events.items():
        if not items:
            continue
        items.sort(key=lambda x: x.date)
        prev_qty = None
        zero_start = None
        zero_start_doc = None
        for event in items:
            qty = event.qty
            if prev_qty is None:
                prev_qty = qty
                if qty <= 0:
                    zero_start = event.date
                    zero_start_doc = event.doc
                continue

            if prev_qty > 0 and qty <= 0 and zero_start is None:
                zero_start = event.date
                zero_start_doc = event.doc
            elif prev_qty <= 0 and qty > 0 and zero_start is not None:
                days = (event.date - zero_start).days
                rows.append(
                    {
                        "SKU": sku,
                        "Artikal": article_names.get(sku, ""),
                        "Zero od": zero_start.date().isoformat(),
                        "Zero do": event.date.date().isoformat(),
                        "Dani bez zalihe": days,
                        "Dokument start": zero_start_doc,
                        "Dokument kraj": event.doc,
                    }
                )
                zero_start = None
                zero_start_doc = None
            prev_qty = qty

        if zero_start is not None:
            rows.append(
                {
                    "SKU": sku,
                    "Artikal": article_names.get(sku, ""),
                    "Zero od": zero_start.date().isoformat(),
                    "Zero do": "",
                    "Dani bez zalihe": "",
                    "Dokument start": zero_start_doc,
                    "Dokument kraj": "",
                }
            )

    out_path = output_dir / "kartice_zero_intervali.csv"
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "SKU",
                "Artikal",
                "Zero od",
                "Zero do",
                "Dani bez zalihe",
                "Dokument start",
                "Dokument kraj",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return out_path


def extract_kartica_events_and_summary(pdf_path: Path, output_dir: Path) -> tuple[Path, Path, Path]:
    parsed = parse_kartica_pdf(pdf_path)
    if not parsed:
        out_events = output_dir / "kartice_events.csv"
        out_summary = output_dir / "kartice_sku_summary.csv"
        out_zero = output_dir / "kartice_zero_intervali.csv"
        pd.DataFrame().to_csv(out_events, index=False)
        pd.DataFrame().to_csv(out_summary, index=False)
        pd.DataFrame().to_csv(out_zero, index=False)
        return out_events, out_summary, out_zero

    out_events = output_dir / "kartice_events.csv"
    event_rows = []
    for r in parsed:
        smer = ""
        if r.prijem_kolicina and not r.izdavanje_kolicina:
            smer = "PRIJEM"
        elif r.izdavanje_kolicina and not r.prijem_kolicina:
            smer = "POVRAT" if r.izdavanje_kolicina < 0 else "IZDAVANJE"
        elif r.prijem_kolicina or r.izdavanje_kolicina:
            smer = "MIXED"
        event_rows.append(
            {
                "SKU": r.sku,
                "Artikal": r.article,
                "Datum": r.date.date().isoformat(),
                "Broj": r.doc,
                "Tip": r.doc_tip,
                "Smer": smer,
                "Opis": r.opis,
                "Referenca": r.referenca,
                "Prijem kolicina": r.prijem_kolicina,
                "Prijem vrednost": r.prijem_vrednost,
                "Izdavanje kolicina": r.izdavanje_kolicina,
                "Izdavanje vrednost": r.izdavanje_vrednost,
                "Cena": r.cena,
                "Stanje zaliha kolicina": r.stanje_kolicina,
                "Stanje zaliha vrednost": r.stanje_vrednost,
                "Delta kolicina": r.delta_kolicina,
            }
        )
    pd.DataFrame(event_rows).to_csv(out_events, index=False, encoding="utf-8")

    zero_csv = write_zero_intervals_from_parsed(parsed, output_dir)
    zero_intervals = _load_zero_intervals(zero_csv)

    df = pd.DataFrame(event_rows)
    df["Datum_dt"] = df["Datum"].apply(_parse_date)
    df = df.loc[df["Datum_dt"].notna()]

    def _first_receipt_date(group: pd.DataFrame) -> str:
        receipt = group.loc[pd.to_numeric(group["Prijem kolicina"], errors="coerce").fillna(0) > 0]
        if receipt.empty:
            return ""
        dt = receipt["Datum_dt"].min().date()
        return dt.isoformat()

    summary_rows = []
    for sku, group in df.groupby("SKU"):
        group = group.sort_values("Datum_dt")
        first_receipt = _first_receipt_date(group)
        first_date = group["Datum_dt"].min().date()
        last_date = group["Datum_dt"].max().date()
        total_days = (last_date - first_date).days + 1
        unavailable_days = _overlap_days(
            datetime.combine(first_date, datetime.min.time()),
            datetime.combine(last_date, datetime.min.time()),
            zero_intervals.get(sku, []),
        )
        available_days = max(total_days - unavailable_days, 0)

        prijem_total = float(pd.to_numeric(group["Prijem kolicina"], errors="coerce").fillna(0).sum())
        izd_total_pos = float(
            pd.to_numeric(group["Izdavanje kolicina"], errors="coerce")
            .fillna(0)
            .clip(lower=0)
            .sum()
        )
        izd_total_neg = float(
            -pd.to_numeric(group["Izdavanje kolicina"], errors="coerce")
            .fillna(0)
            .clip(upper=0)
            .sum()
        )
        prodato_neto = max(izd_total_pos - izd_total_neg, 0.0)
        avg_daily_qty = prodato_neto / available_days if available_days else 0.0
        lost_qty = avg_daily_qty * unavailable_days if unavailable_days else 0.0

        summary_rows.append(
            {
                "SKU": sku,
                "Artikal": (group["Artikal"].iloc[0] if "Artikal" in group.columns else ""),
                "Prvi prijem": first_receipt,
                "Prvi datum": first_date.isoformat(),
                "Zadnji datum": last_date.isoformat(),
                "Ukupno dana": total_days,
                "Dani bez zalihe": int(unavailable_days),
                "Dani dostupno": int(available_days),
                "Nabavljeno": round(prijem_total, 4),
                "Nabavljeno kartica": round(prijem_total, 4),
                "Izdavanje ukupno": round(izd_total_pos, 4),
                "Povrati": round(izd_total_neg, 4),
                "Total prodato": round(prodato_neto, 4),
                "Prosek dnevno": round(avg_daily_qty, 6),
                "Procena izgubljeno": round(lost_qty, 4),
                "Ukupno neto": 0.0,
                "Prosek neto": 0.0,
                "Procjena gubitka neto": 0.0,
            }
        )

    out_summary = output_dir / "kartice_sku_summary.csv"
    pd.DataFrame(summary_rows).to_csv(out_summary, index=False, encoding="utf-8")
    return out_events, out_summary, zero_csv


def _iter_sales_files(sales_path: Path) -> list[Path]:
    if sales_path.is_file():
        return [sales_path]
    if not sales_path.exists():
        return []
    return sorted([p for p in sales_path.glob("*.xlsx") if p.is_file()])


def _iter_receipt_files(receipts_path: Path) -> list[Path]:
    if receipts_path.is_file():
        return [receipts_path]
    if not receipts_path.exists():
        return []
    return sorted([p for p in receipts_path.glob("*.xlsx") if p.is_file()])


def _load_receipt_rows(receipts_path: Path) -> pd.DataFrame:
    frames = []
    for path in _iter_receipt_files(receipts_path):
        xls = pd.ExcelFile(path)
        for sheet in xls.sheet_names:
            df = pd.read_excel(path, sheet_name=sheet)
            if df.empty:
                continue
            norm_cols = {c: _normalize(c) for c in df.columns}
            df = df.rename(columns=norm_cols)
            if "Sifra proizvoda" not in df.columns:
                continue
            df["__source_file"] = path.name
            df["__source_sheet"] = sheet
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def build_receipts_summary(receipts_path: Path, output_dir: Path) -> tuple[Path | None, Path | None]:
    df = _load_receipt_rows(receipts_path)
    if df.empty:
        return None, None

    df["SKU"] = df["Sifra proizvoda"].astype(str).str.strip()
    df = df.loc[df["SKU"].astype(str).str.len() > 0]
    if df.empty:
        return None, None

    def _num(name: str) -> pd.Series:
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce").fillna(0)
        return pd.Series(0, index=df.index)

    df["Poslata kolicina"] = _num("Poslata kolicina")
    df["Pristigla kolicina"] = _num("Pristigla kolicina")
    df["Datum verifikacije"] = df.get("Datum verifikacije", pd.Series(None, index=df.index)).apply(
        _parse_date
    )

    detail_cols = [
        c
        for c in [
            "SKU",
            "Ime proizvoda",
            "Status",
            "Poslata kolicina",
            "Pristigla kolicina",
            "Datum verifikacije",
            "__source_file",
            "__source_sheet",
        ]
        if c in df.columns
    ]
    out_detail = output_dir / "sp_prijemi_detail.csv"
    df[detail_cols].to_csv(out_detail, index=False, encoding="utf-8")

    grp = df.groupby("SKU", dropna=False).agg(
        poslato_sum=("Poslata kolicina", "sum"),
        pristiglo_sum=("Pristigla kolicina", "sum"),
        prvi_verifikovan=("Datum verifikacije", "min"),
        zadnji_verifikovan=("Datum verifikacije", "max"),
    )
    grp = grp.reset_index()
    for col in ("prvi_verifikovan", "zadnji_verifikovan"):
        if col in grp.columns:
            grp[col] = grp[col].apply(lambda v: v.date().isoformat() if isinstance(v, datetime) else "")

    out_summary = output_dir / "sp_prijemi_summary.csv"
    grp.to_csv(out_summary, index=False, encoding="utf-8")
    return out_detail, out_summary


def merge_receipts_into_kartice_summary(kartice_summary: Path, receipts_summary: Path) -> Path:
    if not kartice_summary.exists() or not receipts_summary.exists():
        return kartice_summary

    kdf = pd.read_csv(kartice_summary)
    rdf = pd.read_csv(receipts_summary)
    if kdf.empty or rdf.empty or "SKU" not in kdf.columns or "SKU" not in rdf.columns:
        return kartice_summary

    rdf = rdf.rename(
        columns={
            "poslato_sum": "Poslato (SP prijemi)",
            "pristiglo_sum": "Pristiglo (SP prijemi)",
            "prvi_verifikovan": "Prvi verifikovani prijem",
            "zadnji_verifikovan": "Zadnji verifikovani prijem",
        }
    )
    merged = kdf.merge(rdf, on="SKU", how="left")

    if "Pristiglo (SP prijemi)" in merged.columns:
        sp_val = pd.to_numeric(merged["Pristiglo (SP prijemi)"], errors="coerce")
        if "Nabavljeno" in merged.columns:
            merged["Nabavljeno"] = pd.to_numeric(merged["Nabavljeno"], errors="coerce").fillna(0)
            merged["Nabavljeno"] = merged["Nabavljeno"].where(sp_val.isna(), sp_val.fillna(0))
        else:
            merged["Nabavljeno"] = sp_val.fillna(0)

    merged.to_csv(kartice_summary, index=False, encoding="utf-8")
    return kartice_summary


def _load_sales_rows(sales_path: Path) -> pd.DataFrame:
    frames = []
    files = _iter_sales_files(sales_path)
    for path in files:
        xls = pd.ExcelFile(path)
        for sheet in xls.sheet_names:
            df = pd.read_excel(path, sheet_name=sheet)
            if df.empty:
                continue
            norm_cols = {c: _normalize(c) for c in df.columns}
            df = df.rename(columns=norm_cols)
            if "Sifra proizvoda" not in df.columns or "Kolicina proizvoda" not in df.columns:
                continue
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _load_zero_intervals(path: Path) -> dict[str, list[tuple[datetime, datetime | None]]]:
    intervals: dict[str, list[tuple[datetime, datetime | None]]] = {}
    if not path.exists():
        return intervals
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sku = (row.get("SKU") or "").strip()
            if not sku:
                continue
            start = _parse_date(row.get("Zero od"))
            end = _parse_date(row.get("Zero do"))
            if start is None:
                continue
            intervals.setdefault(sku, []).append((start, end))
    return intervals


def _overlap_days(
    start: datetime, end: datetime, zero_intervals: list[tuple[datetime, datetime | None]]
) -> int:
    range_start = start
    range_end = end + timedelta(days=1)
    total = 0
    for zero_start, zero_end in zero_intervals:
        z_start = zero_start
        z_end = zero_end + timedelta(days=1) if zero_end else range_end
        overlap_start = max(range_start, z_start)
        overlap_end = min(range_end, z_end)
        if overlap_end > overlap_start:
            total += (overlap_end - overlap_start).days
    return total


def build_sales_availability(
    sales_path: Path, zero_csv: Path, output_dir: Path, status_filter: str | None
) -> Path | None:
    sales = _load_sales_rows(sales_path)
    if sales.empty:
        return None

    if status_filter and "Status" in sales.columns:
        status_norm = sales["Status"].astype(str).apply(_normalize).str.strip().str.lower()
        sales = sales.loc[status_norm == status_filter.strip().lower()]

    sales["SKU"] = sales["Sifra proizvoda"].astype(str).str.strip()
    sales["Kolicina"] = pd.to_numeric(sales["Kolicina proizvoda"], errors="coerce").fillna(0)
    def _num_col(name: str) -> pd.Series:
        if name in sales.columns:
            return pd.to_numeric(sales[name], errors="coerce").fillna(0)
        return pd.Series(0, index=sales.index)

    otkup = _num_col("Otkup proizvoda")
    popust = _num_col("Popust proizvoda")
    sales["Net value"] = (otkup - popust) * sales["Kolicina"]

    date_col = None
    for candidate in ("Datum isporuke", "Datum kreiranja", "Datum"):
        if candidate in sales.columns:
            date_col = candidate
            break
    if date_col is None:
        return None

    sales["Datum"] = sales[date_col].apply(_parse_date)
    sales = sales.loc[sales["Datum"].notna()]
    if sales.empty:
        return None

    zero_intervals = _load_zero_intervals(zero_csv)
    rows = []
    for sku, group in sales.groupby("SKU"):
        total_qty = group["Kolicina"].sum()
        first_date = group["Datum"].min().date()
        last_date = group["Datum"].max().date()
        total_days = (last_date - first_date).days + 1
        unavailable_days = _overlap_days(
            datetime.combine(first_date, datetime.min.time()),
            datetime.combine(last_date, datetime.min.time()),
            zero_intervals.get(sku, []),
        )
        available_days = max(total_days - unavailable_days, 0)
        avg_daily_qty = total_qty / available_days if available_days else 0
        lost_qty = avg_daily_qty * unavailable_days if unavailable_days else 0
        total_net = group["Net value"].sum()
        avg_net = total_net / available_days if available_days else 0
        lost_net = avg_net * unavailable_days if unavailable_days else 0
        rows.append(
            {
                "SKU": sku,
                "Total prodato": total_qty,
                "Prvi datum": first_date.isoformat(),
                "Zadnji datum": last_date.isoformat(),
                "Ukupno dana": total_days,
                "Dani bez zalihe": unavailable_days,
                "Dani dostupno": available_days,
                "Prosek dnevno": round(avg_daily_qty, 4),
                "Ukupno neto": round(total_net, 2),
                "Prosek neto": round(avg_net, 4),
                "Procjena gubitka neto": round(lost_net, 2),
                "Procena izgubljeno": round(lost_qty, 4),
            }
        )

    out_path = output_dir / "prodaja_avg_i_gubitak.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    return out_path


def write_top_lists(agg_path: Path, sales_path: Path, output_dir: Path) -> list[Path]:
    outputs = []
    if agg_path.exists():
        df = pd.read_csv(agg_path)
        if "marza_sum" in df.columns:
            top_marza = df.sort_values("marza_sum", ascending=False).head(20)
            out_marza = output_dir / "top_marza.csv"
            top_marza.to_csv(out_marza, index=False)
            outputs.append(out_marza)

    if sales_path.exists():
        df = pd.read_csv(sales_path)
        if "Dani bez zalihe" in df.columns:
            top_zero = df.sort_values("Dani bez zalihe", ascending=False).head(20)
            out_zero = output_dir / "top_nedostupnost.csv"
            top_zero.to_csv(out_zero, index=False)
            outputs.append(out_zero)
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract kalkulacije and kartice intervals into CSVs."
    )
    parser.add_argument(
        "--excel",
        type=Path,
        default=Path(
            r"C:\Users\HOME\Desktop\Srbija1.0 aplikacija\Kalkulacije_kartice_art\Minimax111.xlsx"
        ),
        help="Path to Minimax Excel export with kalkulacije",
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        default=Path(
            r"C:\Users\HOME\Desktop\Srbija1.0 aplikacija\Kalkulacije_kartice_art\Kartica_20260119_070347.pdf"
        ),
        help="Path to kartica PDF export",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(
            r"C:\Users\HOME\Desktop\Srbija1.0 aplikacija\Kalkulacije_kartice_art\izlaz"
        ),
        help="Output directory",
    )
    parser.add_argument(
        "--skip-pdf",
        action="store_true",
        help="Skip PDF parsing for kartice",
    )
    parser.add_argument(
        "--skip-excel",
        action="store_true",
        help="Skip Excel parsing for kalkulacije",
    )
    parser.add_argument(
        "--sales",
        type=Path,
        default=Path(
            r"C:\Users\HOME\Desktop\Srbija1.0 aplikacija\SP-Narudzbe"
        ),
        help="Path to sales Excel file or folder (SP-Narudzbe)",
    )
    parser.add_argument(
        "--sales-status",
        type=str,
        default="Isporuceno",
        help="Filter sales by status (normalized, optional)",
    )
    parser.add_argument(
        "--prijemi",
        type=Path,
        default=Path(r"C:\Users\HOME\Desktop\Srbija1.0 aplikacija\SP Prijemi"),
        help="Path to SP Prijemi Excel file or folder (receipts).",
    )
    parser.add_argument(
        "--skip-prijemi",
        action="store_true",
        help="Skip SP Prijemi parsing",
    )
    parser.add_argument(
        "--build-sp-sales",
        action="store_true",
        help="Build SP sales availability + net loss CSVs (legacy; uses SP-Narudzbe).",
    )
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    detail_path = args.out / "kalkulacije_marza.csv"
    agg_path = args.out / "kalkulacije_marza_agregat.csv"
    if not args.skip_excel:
        detail_path, agg_path = extract_kalkulacije(args.excel, args.out)
        print(f"kalkulacije: {detail_path}")
        print(f"kalkulacije agregat: {agg_path}")

    if not args.skip_pdf:
        events_path, summary_path, zero_path = extract_kartica_events_and_summary(
            args.pdf, args.out
        )
        print(f"kartice events: {events_path}")
        print(f"kartice summary: {summary_path}")
        print(f"kartice zero: {zero_path}")
    else:
        zero_path = args.out / "kartice_zero_intervali.csv"

    if not args.skip_prijemi:
        receipts_detail, receipts_summary = build_receipts_summary(args.prijemi, args.out)
        if receipts_detail and receipts_summary:
            print(f"sp prijemi detail: {receipts_detail}")
            print(f"sp prijemi summary: {receipts_summary}")
            if not args.skip_pdf:
                merge_receipts_into_kartice_summary(summary_path, receipts_summary)
        else:
            print("sp prijemi: nema podataka")

    if args.build_sp_sales:
        sales_out = build_sales_availability(
            args.sales, zero_path, args.out, args.sales_status
        )
        if sales_out:
            print(f"prodaja avg: {sales_out}")
            top_outputs = write_top_lists(agg_path, sales_out, args.out)
            for path in top_outputs:
                print(f"top: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
