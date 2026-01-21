import argparse
import csv
import hashlib
import math
import statistics
import json
import re
import sqlite3
import shutil
import subprocess
import sys
import unicodedata
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd


DB_PATH = Path(__file__).with_suffix(".db")

EXTRACT_SCRIPT = Path(__file__).with_name("extract_kalkulacije_kartice.py")

KALKULACIJE_DIR = Path("Kalkulacije_kartice_art")
KALKULACIJE_OUT_DIR = KALKULACIJE_DIR / "izlaz"
ZERO_INTERVAL_CSV = KALKULACIJE_OUT_DIR / "kartice_zero_intervali.csv"
KALKULACIJE_AGG_CSV = KALKULACIJE_OUT_DIR / "kalkulacije_marza_agregat.csv"
KALKULACIJE_DETAIL_CSV = KALKULACIJE_OUT_DIR / "kalkulacije_marza.csv"
PRODAJA_STATS_CSV = KALKULACIJE_OUT_DIR / "prodaja_avg_i_gubitak.csv"

ZERO_SOURCE = "kartice_zero_intervals"
KALKULACIJE_SOURCE = "kartice_kalkulacije"
KALKULACIJE_DETAIL_SOURCE = "kartice_kalkulacije_detail"
PRODAJA_SOURCE = "kartice_prodaja_stats"
SOURCE_LABELS = {
    ZERO_SOURCE: "Zero intervali",
    KALKULACIJE_SOURCE: "Kalkulacije",
    KALKULACIJE_DETAIL_SOURCE: "Detalji kalkulacija",
    PRODAJA_SOURCE: "Prodaja",
}

SHEET_SP_ORDERS = "Porud\u017ebine"
SHEET_SP_PAYMENTS = "Knji\u017eenje kupaca"
SHEET_MINIMAX = "Minimax"

COL = {
    "client": "Klijent",
    "tracking": "Kod po\u0161iljke",
    "sp_order_no": "Broj porud\u017ebine",
    "woo_order_no": "Klijent broj porud\u017ebine",
    "location": "Lokacija",
    "customer_code": "\u0160ifra kupca",
    "customer_name": "Ime",
    "city": "Grad",
    "address": "Adresa",
    "postal_code": "Po\u0161tanski broj",
    "phone": "Broj telefona",
    "email": "Broj telefona.1",
    "note": "Napomena",
    "product_code": "\u0160ifra proizvoda",
    "qty": "Koli\u010dina proizvoda",
    "cod_amount": "Otkup proizvoda",
    "advance_amount": "Avansni iznos proizvoda",
    "discount": "Popust proizvoda",
    "discount_type": "Tip popusta proizvoda",
    "addon_cod": "Otkup dodatak",
    "addon_advance": "Avansni iznos dodatka",
    "extra_discount": "Popust",
    "extra_discount_type": "Tip popusta",
    "status": "Status",
    "created_at": "Datum kreiranja",
    "picked_up_at": "Datum preuzimanja",
    "delivered_at": "Datum isporuke",
    # SP-Uplate
    "payment_amount": "Iznos",
    "payment_order_status": "Status porud\u017ebine",
    "payment_client_status": "Status klijenta",
    "payment_customer_name": "Ime kupca",
    # Minimax
    "mm_number": "Broj",
    "mm_customer": "Kupac",
    "mm_country": "Dr\u017eava stranke",
    "mm_date": "Datum",
    "mm_due_date": "Dospe\u0107e",
    "mm_revenue": "Prihod",
    "mm_amount_local": "Iznos u NJ",
    "mm_amount_due": "Iznos za pla\u0107anje",
    "mm_analytics": "Analitika",
    "mm_turnover": "Promet",
    "mm_account": "Ra\u010dun",
    "mm_basis": "Osnova za ra\u010dun",
    "mm_note": "Napomene",
    "mm_payment_amount": "Iznos pla\u0107anja",
    "mm_open_amount": "Otvoreno",
}

SETTINGS_PATH = Path(__file__).with_name("srb_settings.json")
CUSTOMER_KEY_VERSION = "2"

DEFAULT_PREFIX_MAP = {
    "AF-": "Afro rep",
    "RR-": "Ravni rep",
    "OPK-": "Repovi OPK",
    "AR-": "Ariana repovi",
    "KRR-": "Kratki repovi",
    "KRO-": "Kratki repovi",
    "KRA-": "Kratki repovi",
    "TRK-": "Repovi trakica",
    "DR-": "Dugi repovi",
    "U-": "U klipse",
    "BD-": "Blowdry klipse",
    "BDR-": "Blowdry repovi",
    "EKS-": "Ekstenzije",
    "EKSOPK": "Ekstenzije OPK",
    "SIS-": "Siske",
    "P0": "Klasicne perike",
    "PR": "Premium perike",
}

DEFAULT_CUSTOM_SKU_LIST = []
DEFAULT_SKU_CATEGORY_OVERRIDES = {}


def _normalize_sku_list(values):
    result = []
    for v in values or []:
        if isinstance(v, str):
            v = v.strip().upper()
            if v:
                result.append(v)
    return result


def _normalize_prefix_map(values):
    result = {}
    for k, v in (values or {}).items():
        if not isinstance(k, str) or not isinstance(v, str):
            continue
        key = k.strip().upper()
        val = v.strip()
        if key and val:
            result[key] = val
    return result


def _normalize_overrides(values):
    result = {}
    for k, v in (values or {}).items():
        if not isinstance(k, str) or not isinstance(v, str):
            continue
        key = k.strip().upper()
        val = v.strip()
        if key and val:
            result[key] = val
    return result


def load_category_settings():
    prefix = DEFAULT_PREFIX_MAP.copy()
    custom_list = list(DEFAULT_CUSTOM_SKU_LIST)
    overrides = DEFAULT_SKU_CATEGORY_OVERRIDES.copy()
    if SETTINGS_PATH.exists():
        try:
            data = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
            prefix.update(_normalize_prefix_map(data.get("prefix_map")))
            custom_list = _normalize_sku_list(data.get("custom_skus", custom_list))
            overrides = _normalize_overrides(
                data.get("sku_category_overrides", overrides)
            )
        except Exception:
            pass
    return prefix, custom_list, overrides


def save_category_settings() -> None:
    data = {
        "prefix_map": prefix_map,
        "custom_skus": sorted(CUSTOM_SKU_SET),
        "sku_category_overrides": SKU_CATEGORY_OVERRIDES,
    }
    SETTINGS_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=True), encoding="utf-8")


def load_app_settings() -> dict:
    if SETTINGS_PATH.exists():
        try:
            return json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_app_settings(values: dict) -> None:
    data = load_app_settings()
    data.update(values)
    SETTINGS_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=True), encoding="utf-8")


prefix_map, CUSTOM_SKU_LIST, SKU_CATEGORY_OVERRIDES = load_category_settings()
CUSTOM_SKU_SET = {s.upper() for s in CUSTOM_SKU_LIST}


def sifra_to_prefix(sifra: str) -> str:
    if not isinstance(sifra, str):
        return ""
    sifra = sifra.strip().upper()
    candidates = [p for p in prefix_map.keys() if sifra.startswith(p)]
    if not candidates:
        return ""
    return max(candidates, key=len)


def kategorija_za_sifru(sifra: str, allow_custom: bool = True) -> str:
    if not isinstance(sifra, str):
        return "Ostalo"
    sku = sifra.strip().upper()
    if sku in SKU_CATEGORY_OVERRIDES:
        return SKU_CATEGORY_OVERRIDES[sku]
    if allow_custom and sku in CUSTOM_SKU_SET:
        return "Custom"
    pref = sifra_to_prefix(sifra)
    return prefix_map.get(pref, "Ostalo")


def add_category_prefix(prefix: str, name: str) -> None:
    prefix = (prefix or "").strip().upper()
    name = (name or "").strip()
    if not prefix or not name:
        raise ValueError("Prefix i naziv su obavezni.")
    prefix_map[prefix] = name
    save_category_settings()


def add_sku_category_override(sku: str, category: str) -> None:
    sku = (sku or "").strip().upper()
    category = (category or "").strip()
    if not sku or not category:
        raise ValueError("SKU i kategorija su obavezni.")
    SKU_CATEGORY_OVERRIDES[sku] = category
    save_category_settings()


def add_custom_sku(sku: str) -> None:
    sku = (sku or "").strip().upper()
    if not sku:
        raise ValueError("SKU je obavezan.")
    CUSTOM_SKU_SET.add(sku)
    save_category_settings()


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS import_runs (
  id INTEGER PRIMARY KEY,
  source TEXT NOT NULL,
  filename TEXT NOT NULL,
  file_hash TEXT NOT NULL,
  imported_at TEXT NOT NULL,
  row_count INTEGER NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_import_runs_file_hash
  ON import_runs(file_hash);

CREATE TABLE IF NOT EXISTS app_state (
  key TEXT PRIMARY KEY,
  value TEXT
);

CREATE TABLE IF NOT EXISTS orders (
  id INTEGER PRIMARY KEY,
  sp_order_no TEXT NOT NULL,
  woo_order_no TEXT,
  client_code TEXT,
  tracking_code TEXT,
  customer_code TEXT,
  customer_name TEXT,
  city TEXT,
  address TEXT,
  postal_code TEXT,
  phone TEXT,
  email TEXT,
  customer_key TEXT,
  note TEXT,
  location TEXT,
  status TEXT,
  created_at TEXT,
  picked_up_at TEXT,
  delivered_at TEXT,
  import_run_id INTEGER,
  FOREIGN KEY(import_run_id) REFERENCES import_runs(id)
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_orders_sp_order_no
  ON orders(sp_order_no);

CREATE INDEX IF NOT EXISTS idx_orders_tracking
  ON orders(tracking_code);

CREATE INDEX IF NOT EXISTS idx_orders_dates
  ON orders(picked_up_at, delivered_at);

CREATE TABLE IF NOT EXISTS order_status_history (
  id INTEGER PRIMARY KEY,
  order_id INTEGER NOT NULL,
  status TEXT NOT NULL,
  status_at TEXT NOT NULL,
  source TEXT,
  note TEXT,
  FOREIGN KEY(order_id) REFERENCES orders(id)
);

CREATE INDEX IF NOT EXISTS idx_order_status_order
  ON order_status_history(order_id);

CREATE TABLE IF NOT EXISTS order_items (
  id INTEGER PRIMARY KEY,
  order_id INTEGER NOT NULL,
  product_code TEXT,
  qty REAL,
  cod_amount REAL,
  advance_amount REAL,
  discount REAL,
  discount_type TEXT,
  addon_cod REAL,
  addon_advance REAL,
  extra_discount REAL,
  extra_discount_type TEXT,
  FOREIGN KEY(order_id) REFERENCES orders(id)
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_order_items_unique
  ON order_items(order_id, product_code, qty, cod_amount);

CREATE INDEX IF NOT EXISTS idx_order_items_order
  ON order_items(order_id);

CREATE INDEX IF NOT EXISTS idx_order_items_product
  ON order_items(product_code);

CREATE TABLE IF NOT EXISTS payments (
  id INTEGER PRIMARY KEY,
  sp_order_no TEXT NOT NULL,
  client_code TEXT,
  customer_code TEXT,
  customer_name TEXT,
  amount REAL,
  order_status TEXT,
  client_status TEXT,
  import_run_id INTEGER,
  FOREIGN KEY(import_run_id) REFERENCES import_runs(id)
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_payments_dedupe
  ON payments(sp_order_no, amount, client_status);

CREATE INDEX IF NOT EXISTS idx_payments_sp_order_no
  ON payments(sp_order_no);

CREATE TABLE IF NOT EXISTS returns (
  id INTEGER PRIMARY KEY,
  sp_order_no TEXT,
  tracking_code TEXT,
  customer_name TEXT,
  phone TEXT,
  city TEXT,
  status TEXT,
  created_at TEXT,
  picked_up_at TEXT,
  delivered_at TEXT,
  import_run_id INTEGER,
  FOREIGN KEY(import_run_id) REFERENCES import_runs(id)
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_returns_dedupe
  ON returns(sp_order_no, tracking_code);

CREATE INDEX IF NOT EXISTS idx_returns_tracking
  ON returns(tracking_code);

CREATE TABLE IF NOT EXISTS invoices (
  id INTEGER PRIMARY KEY,
  number TEXT,
  customer_name TEXT,
  country TEXT,
  date TEXT,
  due_date TEXT,
  revenue TEXT,
  amount_local REAL,
  amount_due REAL,
  analytics TEXT,
  turnover TEXT,
  account TEXT,
  basis TEXT,
  note TEXT,
  payment_amount REAL,
  open_amount REAL,
  import_run_id INTEGER,
  FOREIGN KEY(import_run_id) REFERENCES import_runs(id)
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_invoices_number
  ON invoices(number);

CREATE INDEX IF NOT EXISTS idx_invoices_date_amount
  ON invoices(date, amount_due);

CREATE TABLE IF NOT EXISTS invoice_matches (
  id INTEGER PRIMARY KEY,
  order_id INTEGER NOT NULL,
  invoice_id INTEGER NOT NULL,
  score INTEGER NOT NULL,
  status TEXT NOT NULL, -- auto, review, needs_invoice
  method TEXT,
  matched_at TEXT NOT NULL,
  FOREIGN KEY(order_id) REFERENCES orders(id),
  FOREIGN KEY(invoice_id) REFERENCES invoices(id)
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_invoice_matches_order
  ON invoice_matches(order_id);

CREATE UNIQUE INDEX IF NOT EXISTS idx_invoice_matches_invoice
  ON invoice_matches(invoice_id);

CREATE TABLE IF NOT EXISTS invoice_candidates (
  id INTEGER PRIMARY KEY,
  order_id INTEGER NOT NULL,
  invoice_id INTEGER NOT NULL,
  score INTEGER NOT NULL,
  detail TEXT,
  method TEXT,
  created_at TEXT NOT NULL,
  FOREIGN KEY(order_id) REFERENCES orders(id),
  FOREIGN KEY(invoice_id) REFERENCES invoices(id)
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_invoice_candidates_pair
  ON invoice_candidates(order_id, invoice_id);

CREATE TABLE IF NOT EXISTS order_flags (
  id INTEGER PRIMARY KEY,
  order_id INTEGER NOT NULL,
  flag TEXT NOT NULL, -- needs_invoice
  note TEXT,
  created_at TEXT NOT NULL,
  FOREIGN KEY(order_id) REFERENCES orders(id)
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_order_flags_unique
  ON order_flags(order_id, flag);

CREATE TABLE IF NOT EXISTS invoice_storno (
  id INTEGER PRIMARY KEY,
  storno_invoice_id INTEGER NOT NULL,
  original_invoice_id INTEGER NOT NULL,
  storno_amount REAL,
  original_amount REAL,
  remaining_open REAL,
  is_partial INTEGER NOT NULL,
  created_at TEXT NOT NULL,
  FOREIGN KEY(storno_invoice_id) REFERENCES invoices(id),
  FOREIGN KEY(original_invoice_id) REFERENCES invoices(id)
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_invoice_storno_unique
  ON invoice_storno(storno_invoice_id);

CREATE TABLE IF NOT EXISTS action_log (
  id INTEGER PRIMARY KEY,
  action TEXT NOT NULL,
  ref_type TEXT NOT NULL,
  ref_id INTEGER NOT NULL,
  note TEXT,
  created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_action_log_ref
  ON action_log(ref_type, ref_id);

CREATE TABLE IF NOT EXISTS minimax_items (
  id INTEGER PRIMARY KEY,
  sku TEXT NOT NULL,
  name TEXT,
  unit TEXT,
  mass_kg REAL,
  stock REAL,
  opening_qty REAL,
  opening_purchase_value REAL,
  opening_sales_value REAL,
  incoming_qty REAL,
  incoming_purchase_value REAL,
  incoming_sales_value REAL,
  outgoing_qty REAL,
  outgoing_purchase_value REAL,
  outgoing_sales_value REAL,
  stock_2 REAL,
  closing_qty REAL,
  closing_purchase_value REAL,
  closing_sales_value REAL,
  updated_at TEXT,
  import_run_id INTEGER,
  FOREIGN KEY(import_run_id) REFERENCES import_runs(id)
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_minimax_items_sku
  ON minimax_items(sku);

CREATE TABLE IF NOT EXISTS bank_transactions (
  id INTEGER PRIMARY KEY,
  fitid TEXT NOT NULL,
  stmt_number TEXT,
  benefit TEXT,
  dtposted TEXT,
  amount REAL,
  purpose TEXT,
  purposecode TEXT,
  payee_name TEXT,
  payee_city TEXT,
  payee_acctid TEXT,
  payee_bankid TEXT,
  payee_bankname TEXT,
  refnumber TEXT,
  payeerefnumber TEXT,
  urgency TEXT,
  fee REAL,
  import_run_id INTEGER,
  FOREIGN KEY(import_run_id) REFERENCES import_runs(id)
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_bank_fitid
  ON bank_transactions(fitid);

CREATE INDEX IF NOT EXISTS idx_bank_dtposted
  ON bank_transactions(dtposted);

CREATE INDEX IF NOT EXISTS idx_bank_payee
  ON bank_transactions(payee_name);

CREATE TABLE IF NOT EXISTS bank_refunds (
  id INTEGER PRIMARY KEY,
  bank_txn_id INTEGER NOT NULL,
  invoice_no TEXT,
  invoice_no_digits TEXT,
  invoice_no_source TEXT,
  reason TEXT,
  created_at TEXT NOT NULL,
  FOREIGN KEY(bank_txn_id) REFERENCES bank_transactions(id)
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_bank_refunds_txn
  ON bank_refunds(bank_txn_id);

CREATE TABLE IF NOT EXISTS bank_matches (
  id INTEGER PRIMARY KEY,
  bank_txn_id INTEGER NOT NULL,
  match_type TEXT NOT NULL, -- sp_payment | storno
  ref_id INTEGER NOT NULL,
  score INTEGER NOT NULL,
  method TEXT,
  matched_at TEXT NOT NULL,
  FOREIGN KEY(bank_txn_id) REFERENCES bank_transactions(id)
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_bank_matches_txn
  ON bank_matches(bank_txn_id);

CREATE TABLE IF NOT EXISTS tracking_events (
  id INTEGER PRIMARY KEY,
  tracking_code TEXT NOT NULL,
  status_time TEXT,
  status_value TEXT,
  fetched_at TEXT NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_tracking_events_unique
  ON tracking_events(tracking_code, status_time, status_value);

  CREATE TABLE IF NOT EXISTS tracking_summary (
    tracking_code TEXT PRIMARY KEY,
    received_at TEXT,
    first_out_for_delivery_at TEXT,
    delivery_attempts INTEGER,
    failure_reasons TEXT,
    returned_at TEXT,
    days_to_first_attempt REAL,
    has_attempt_before_return INTEGER,
    has_returned INTEGER,
    anomalies TEXT,
    last_status TEXT,
    last_status_at TEXT,
    last_fetched_at TEXT NOT NULL
  );
  
  CREATE TABLE IF NOT EXISTS task_progress (
    task TEXT PRIMARY KEY,
    total INTEGER NOT NULL,
    processed INTEGER NOT NULL,
    updated_at TEXT NOT NULL
  );

  CREATE TABLE IF NOT EXISTS cartice_zero_intervals (
    id INTEGER PRIMARY KEY,
    sku TEXT NOT NULL,
    artikal TEXT,
    zero_from TEXT NOT NULL,
    zero_to TEXT,
    days_without INTEGER,
    document_start TEXT,
    document_end TEXT,
    import_run_id INTEGER,
    FOREIGN KEY(import_run_id) REFERENCES import_runs(id),
    UNIQUE(sku, zero_from, document_start)
  );

  CREATE INDEX IF NOT EXISTS idx_cartice_zero_sku
    ON cartice_zero_intervals(sku);

  CREATE TABLE IF NOT EXISTS cartice_zero_meta (
    sku TEXT PRIMARY KEY,
    last_zero_to TEXT,
    last_document TEXT
  );

  CREATE TABLE IF NOT EXISTS cartice_receipts (
    id INTEGER PRIMARY KEY,
    sku TEXT NOT NULL,
    document TEXT NOT NULL,
    quantity REAL,
    updated_at TEXT,
    UNIQUE(sku, document)
  );

  CREATE INDEX IF NOT EXISTS idx_cartice_receipts
    ON cartice_receipts(sku, document);

  CREATE TABLE IF NOT EXISTS cartice_kalkulacije (
    id INTEGER PRIMARY KEY,
    sku TEXT NOT NULL,
    artikal TEXT,
    kolicina_sum REAL,
    nabavna_sum REAL,
    prodajna_sum REAL,
    marza_sum REAL,
    doc_count INTEGER,
    first_date TEXT,
    last_date TEXT,
    import_run_id INTEGER,
    FOREIGN KEY(import_run_id) REFERENCES import_runs(id),
    UNIQUE(sku)
  );

  CREATE INDEX IF NOT EXISTS idx_cartice_kalkulacije_sku
    ON cartice_kalkulacije(sku);

  CREATE TABLE IF NOT EXISTS cartice_kalkulacije_detail (
    id INTEGER PRIMARY KEY,
    opis TEXT NOT NULL UNIQUE,
    sku TEXT,
    artikal TEXT,
    datum TEXT,
    broj TEXT,
    kolicina REAL,
    imported_at TEXT NOT NULL,
    import_run_id INTEGER,
    FOREIGN KEY(import_run_id) REFERENCES import_runs(id)
  );

  CREATE INDEX IF NOT EXISTS idx_cartice_kalkulacije_detail_opis
    ON cartice_kalkulacije_detail(opis);

  CREATE TABLE IF NOT EXISTS prodaja_stats (
    id INTEGER PRIMARY KEY,
    sku TEXT NOT NULL,
    artikal TEXT,
    total_sold REAL,
    first_date TEXT,
    last_date TEXT,
    total_days INTEGER,
    days_without INTEGER,
    days_available INTEGER,
    avg_daily REAL,
    total_net REAL,
    avg_net REAL,
    lost_net REAL,
    lost_qty REAL,
    imported_at TEXT,
    import_run_id INTEGER,
    FOREIGN KEY(import_run_id) REFERENCES import_runs(id),
    UNIQUE(sku)
  );

  CREATE INDEX IF NOT EXISTS idx_prodaja_stats_sku
    ON prodaja_stats(sku);
"""


def connect_db(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA_SQL)
    ensure_column(conn, "invoice_candidates", "detail", "TEXT")
    ensure_column(conn, "orders", "customer_key", "TEXT")
    ensure_column(conn, "minimax_items", "updated_at", "TEXT")
    ensure_column(conn, "tracking_summary", "last_status", "TEXT")
    ensure_column(conn, "tracking_summary", "last_status_at", "TEXT")
    conn.commit()


def ensure_column(conn: sqlite3.Connection, table: str, column: str, col_type: str) -> None:
    cols = conn.execute(f"PRAGMA table_info({table})").fetchall()
    if any(row[1] == column for row in cols):
        return
    conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [row for row in reader]


def _normalize_numeric_literal(text: str) -> str:
    value = text.replace(" ", "")
    if "," in value and "." in value:
        value = value.replace(".", "").replace(",", ".")
    elif "," in value:
        value = value.replace(",", ".")
    return value


def _parse_float(value: str | None, default: float | None = 0.0) -> float | None:
    if value is None:
        return default
    text = str(value).strip()
    if not text:
        return default
    try:
        normalized = _normalize_numeric_literal(text)
        return float(normalized)
    except ValueError:
        return default


def _parse_int(value: str | None, default: int | None = 0) -> int | None:
    if value is None:
        return default
    text = str(value).strip()
    if not text:
        return default
    float_val = _parse_float(text, default=None)
    if float_val is None:
        return default
    try:
        return int(float_val)
    except (ValueError, TypeError):
        return default


def _record_import_run(
    conn: sqlite3.Connection,
    source: str,
    path: Path,
    file_hash_value: str,
    row_count: int,
) -> int | None:
    try:
        cur = conn.execute(
            "INSERT INTO import_runs (source, filename, file_hash, imported_at, row_count) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                source,
                str(path),
                file_hash_value,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                row_count,
            ),
        )
    except sqlite3.IntegrityError:
        return None
    return cur.lastrowid


def _run_cartice_import(
    conn: sqlite3.Connection,
    path: Path,
    source: str,
    apply_fn,
) -> dict[str, str | int]:
    if not path.exists():
        return {"source": source, "status": "missing", "path": str(path)}
    digest = file_hash(path)
    existing = conn.execute(
        "SELECT id FROM import_runs WHERE source = ? AND file_hash = ?",
        (source, digest),
    ).fetchone()
    if existing:
        return {"source": source, "status": "skipped"}
    rows = _read_csv_rows(path)
    import_id = _record_import_run(conn, source, path, digest, len(rows))
    if import_id is None:
        return {"source": source, "status": "skipped"}
    inserted = apply_fn(conn, rows, import_id)
    conn.commit()
    return {"source": source, "status": "imported", "rows": inserted}


def _doc_order_key(doc: str | None) -> tuple[int, ...]:
    if not doc:
        return ()
    digits = [int(value) for value in re.findall(r"\\d+", doc) if value.isdigit()]
    return tuple(digits)


def _load_zero_meta(conn: sqlite3.Connection) -> dict[str, tuple[str | None, str | None]]:
    rows = conn.execute(
        "SELECT sku, last_zero_to, last_document FROM cartice_zero_meta"
    ).fetchall()
    return {sku: (last_zero_to, last_document) for sku, last_zero_to, last_document in rows}


def _load_receipts(conn: sqlite3.Connection) -> dict[tuple[str, str], float | None]:
    rows = conn.execute("SELECT sku, document, quantity FROM cartice_receipts").fetchall()
    result: dict[tuple[str, str], float | None] = {}
    for sku, document, quantity in rows:
        if not document:
            continue
        result[(sku, document)] = float(quantity) if quantity is not None else None
    return result


def _receipt_quantity_from_detail(conn: sqlite3.Connection, sku: str, document: str) -> float | None:
    row = conn.execute(
        "SELECT SUM(kolicina) FROM cartice_kalkulacije_detail WHERE sku = ? AND broj = ?",
        (sku, document),
    ).fetchone()
    value = row[0] if row else None
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _update_receipt_record(
    conn: sqlite3.Connection, sku: str, document: str, quantity: float | None
) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn.execute(
        "INSERT INTO cartice_receipts (sku, document, quantity, updated_at) "
        "VALUES (?, ?, ?, ?) "
        "ON CONFLICT(sku, document) DO UPDATE SET "
        "quantity = excluded.quantity, updated_at = excluded.updated_at",
        (sku, document, quantity, now),
    )


def _should_insert_zero_interval(
    meta_entry: tuple[str | None, str | None] | None,
    candidate_date: date | None,
    candidate_doc_key: tuple[int, ...],
) -> bool:
    if meta_entry is None:
        return True
    meta_date_str, meta_doc = meta_entry
    meta_date = normalize_date(meta_date_str) if meta_date_str else None
    meta_doc_key = _doc_order_key(meta_doc)
    if candidate_date and meta_date:
        if candidate_date < meta_date:
            return False
        if candidate_date > meta_date:
            return True
        return candidate_doc_key > meta_doc_key
    if candidate_date and not meta_date:
        return True
    if not candidate_date and meta_date:
        return candidate_doc_key > meta_doc_key
    return candidate_doc_key > meta_doc_key


def _apply_zero_intervals(conn: sqlite3.Connection, rows: list[dict[str, str]], import_run_id: int) -> int:
    cursor = conn.cursor()
    meta_map = _load_zero_meta(conn)
    receipt_map = _load_receipts(conn)
    insert_sql = (
        "INSERT OR IGNORE INTO cartice_zero_intervals "
        "(sku, artikal, zero_from, zero_to, days_without, document_start, document_end, import_run_id) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
    )
    inserted = 0
    meta_updates: dict[str, tuple[date | None, str | None]] = {}
    for row in rows:
        sku = (row.get("SKU") or "").strip()
        zero_from = (row.get("Zero od") or "").strip()
        zero_to_raw = (row.get("Zero do") or "").strip()
        zero_to = normalize_date(zero_to_raw) if zero_to_raw else None
        if not sku or not zero_from:
            continue
        candidate_date = zero_to or normalize_date(zero_from)
        doc_end = (row.get("Dokument kraj") or "").strip()
        doc_key = _doc_order_key(doc_end)
        if not _should_insert_zero_interval(meta_map.get(sku), candidate_date, doc_key):
            continue
        receipt_qty = None
        if doc_end:
            existing_qty = receipt_map.get((sku, doc_end))
            receipt_qty = _receipt_quantity_from_detail(conn, sku, doc_end)
            skip_due_to_doc = (
                existing_qty is not None
                and (receipt_qty is None or math.isclose(existing_qty, receipt_qty, rel_tol=1e-6))
            )
            if skip_due_to_doc:
                continue
        values = (
            sku,
            (row.get("Artikal") or "").strip(),
            zero_from,
            zero_to_raw or None,
            _parse_int(row.get("Dani bez zalihe"), default=None),
            (row.get("Dokument start") or "").strip(),
            doc_end,
            import_run_id,
        )
        cursor.execute(insert_sql, values)
        if cursor.rowcount:
            inserted += 1
            meta_updates[sku] = (candidate_date, doc_end)
            meta_map[sku] = (
                candidate_date.strftime("%Y-%m-%d") if candidate_date else None,
                doc_end,
            )
            if doc_end:
                _update_receipt_record(conn, sku, doc_end, receipt_qty)
                receipt_map[(sku, doc_end)] = receipt_qty
    for sku, (date_obj, doc_text) in meta_updates.items():
        date_text = date_obj.strftime("%Y-%m-%d") if date_obj else None
        conn.execute(
            "INSERT INTO cartice_zero_meta (sku, last_zero_to, last_document) "
            "VALUES (?, ?, ?) "
            "ON CONFLICT(sku) DO UPDATE SET last_zero_to = excluded.last_zero_to, "
            "last_document = excluded.last_document",
            (sku, date_text, doc_text),
        )
    return inserted


def _apply_kalkulacije(conn: sqlite3.Connection, rows: list[dict[str, str]], import_run_id: int) -> int:
    conn.execute("DELETE FROM cartice_kalkulacije")
    parsed = []
    for row in rows:
        sku = (row.get("SKU") or "").strip()
        if not sku:
            continue
        parsed.append(
            (
                sku,
                (row.get("Artikal") or "").strip(),
                _parse_float(row.get("kolicina_sum")) or 0.0,
                _parse_float(row.get("nabavna_sum")) or 0.0,
                _parse_float(row.get("prodajna_sum")) or 0.0,
                _parse_float(row.get("marza_sum")) or 0.0,
                _parse_int(row.get("doc_count")) or 0,
                (row.get("first_date") or "").strip(),
                (row.get("last_date") or "").strip(),
                import_run_id,
            )
        )
    if parsed:
        conn.executemany(
            "INSERT INTO cartice_kalkulacije "
            "(sku, artikal, kolicina_sum, nabavna_sum, prodajna_sum, marza_sum, doc_count, "
            "first_date, last_date, import_run_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            parsed,
        )
    return len(parsed)


def _apply_kalkulacije_detail(conn: sqlite3.Connection, rows: list[dict[str, str]], import_run_id: int) -> int:
    before = conn.execute("SELECT COUNT(*) FROM cartice_kalkulacije_detail").fetchone()[0]
    parsed = []
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for row in rows:
        opis = (row.get("Opis") or "").strip()
        if not opis:
            continue
        parsed.append(
            (
                opis,
                (row.get("SKU") or "").strip(),
                (row.get("Artikal") or "").strip(),
                (row.get("Datum") or "").strip(),
                (row.get("Broj") or "").strip(),
                _parse_float(row.get("Kolicina")) or 0.0,
                timestamp,
                import_run_id,
            )
        )
    if parsed:
        conn.executemany(
            "INSERT OR IGNORE INTO cartice_kalkulacije_detail "
            "(opis, sku, artikal, datum, broj, kolicina, imported_at, import_run_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            parsed,
        )
    after = conn.execute("SELECT COUNT(*) FROM cartice_kalkulacije_detail").fetchone()[0]
    return after - before


def _apply_prodaja_stats(conn: sqlite3.Connection, rows: list[dict[str, str]], import_run_id: int) -> int:
    conn.execute("DELETE FROM prodaja_stats")
    parsed = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for row in rows:
        sku = (row.get("SKU") or "").strip()
        if not sku:
            continue
        parsed.append(
            (
                sku,
                (row.get("Artikal") or "").strip(),
                _parse_float(row.get("Total prodato")) or 0.0,
                (row.get("Prvi datum") or "").strip(),
                (row.get("Zadnji datum") or "").strip(),
                _parse_int(row.get("Ukupno dana")) or 0,
                _parse_int(row.get("Dani bez zalihe")) or 0,
                _parse_int(row.get("Dani dostupno")) or 0,
                _parse_float(row.get("Prosek dnevno")) or 0.0,
                _parse_float(row.get("Ukupno neto")) or 0.0,
                _parse_float(row.get("Prosek neto")) or 0.0,
                _parse_float(row.get("Procjena gubitka neto")) or 0.0,
                _parse_float(row.get("Procena izgubljeno")) or 0.0,
                now,
                import_run_id,
            )
        )
    if parsed:
        conn.executemany(
            "INSERT INTO prodaja_stats "
            "(sku, artikal, total_sold, first_date, last_date, total_days, days_without, days_available, "
            "avg_daily, total_net, avg_net, lost_net, lost_qty, imported_at, import_run_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            parsed,
        )
    return len(parsed)


def _get_last_import_info(conn: sqlite3.Connection, source: str) -> tuple[str, int] | None:
    row = conn.execute(
        "SELECT imported_at, row_count FROM import_runs "
        "WHERE source = ? "
        "ORDER BY imported_at DESC LIMIT 1",
        (source,),
    ).fetchone()
    if not row:
        return None
    return row[0], int(row[1] or 0)


def set_app_state(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO app_state (key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, value),
    )
    conn.commit()


def get_app_state(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute(
        "SELECT value FROM app_state WHERE key = ?",
        (key,),
    ).fetchone()
    if not row:
        return None
    return row[0]


def hash_password(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


_HTTP_OPENER = None
_HTTP_COOKIE_JAR = None


def _http_headers(extra: dict | None = None) -> dict:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "bs-BA,bs;q=0.9,en-US;q=0.8,en;q=0.7",
        "Connection": "keep-alive",
    }
    if extra:
        headers.update(extra)
    return headers


def _get_http_opener():
    global _HTTP_OPENER, _HTTP_COOKIE_JAR
    if _HTTP_OPENER is not None:
        return _HTTP_OPENER
    import http.cookiejar
    import urllib.request

    _HTTP_COOKIE_JAR = http.cookiejar.CookieJar()
    _HTTP_OPENER = urllib.request.build_opener(
        urllib.request.HTTPCookieProcessor(_HTTP_COOKIE_JAR)
    )
    return _HTTP_OPENER


def http_request(
    url: str,
    method: str = "GET",
    data: bytes | None = None,
    headers: dict | None = None,
    timeout: int = 20,
):
    import urllib.request

    req = urllib.request.Request(url, data=data, method=method, headers=_http_headers(headers))
    opener = _get_http_opener()
    return opener.open(req, timeout=timeout)


def fetch_cbbh_rsd_rate(url: str = "https://www.cbbh.ba/CurrencyExchange/") -> float | None:
    rate, _ = fetch_cbbh_rsd_rate_debug(url)
    return rate


def fetch_cbbh_rsd_rate_debug(url: str = "https://www.cbbh.ba/CurrencyExchange/") -> tuple[float | None, str | None]:
    try:
        import urllib.request

        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            html = resp.read().decode("utf-8", errors="ignore")
    except Exception as exc:
        return None, f"HTTP greska: {exc}"

    rows = re.findall(r"<tr[^>]*>.*?</tr>", html, flags=re.S | re.I)
    row = None
    for candidate in rows:
        if "currcircle\">RSD</div>" in candidate and "<td" in candidate:
            row = candidate
            break
    if not row:
        return None, "RSD red nije pronadjen u HTML-u."
    middle_match = re.search(r"middle-column[^>]*>\s*([0-9.,]+)\s*<", row)
    units_match = re.findall(r"tbl-smaller[^>]*tbl-center[^>]*>\s*(\d+)\s*<", row)
    if not middle_match:
        return None, "Nedostaje middle vrijednost u RSD redu."
    if not units_match:
        return None, "Nedostaje units vrijednost u RSD redu."
    try:
        units = float(units_match[0])
        middle = float(middle_match.group(1).replace(",", "."))
        if units <= 0:
            return None, "Units je 0 ili negativan."
        return middle / units, None
    except (TypeError, ValueError) as exc:
        return None, f"Parse greska: {exc}"


def set_task_progress(conn: sqlite3.Connection, task: str, total: int) -> None:
    conn.execute(
        "INSERT INTO task_progress (task, total, processed, updated_at) "
        "VALUES (?, ?, 0, datetime('now')) "
        "ON CONFLICT(task) DO UPDATE SET total = excluded.total, "
        "processed = excluded.processed, updated_at = excluded.updated_at",
        (task, total),
    )
    conn.commit()


def update_task_progress(conn: sqlite3.Connection, task: str, processed: int) -> None:
    conn.execute(
        "UPDATE task_progress SET processed = ?, updated_at = datetime('now') "
        "WHERE task = ?",
        (processed, task),
    )
    conn.commit()


def get_task_progress(conn: sqlite3.Connection, task: str):
    row = conn.execute(
        "SELECT total, processed, updated_at FROM task_progress WHERE task = ?",
        (task,),
    ).fetchone()
    if not row:
        return None
    return {"total": int(row[0]), "processed": int(row[1]), "updated_at": row[2]}


def start_import(conn: sqlite3.Connection, source: str, path: Path, row_count: int) -> int | None:
    digest = file_hash(path)
    existing = conn.execute(
        "SELECT id FROM import_runs WHERE file_hash = ?",
        (digest,),
    ).fetchone()
    if existing:
        return None
    cur = conn.execute(
        "INSERT INTO import_runs (source, filename, file_hash, imported_at, row_count) "
        "VALUES (?, ?, ?, datetime('now'), ?)",
        (source, str(path), digest, row_count),
    )
    conn.commit()
    return int(cur.lastrowid)


def get_or_create_order(conn: sqlite3.Connection, sp_order_no: str, values: dict) -> tuple[int, bool]:
    row = conn.execute(
        "SELECT id FROM orders WHERE sp_order_no = ?",
        (sp_order_no,),
    ).fetchone()
    if row:
        return int(row[0]), False
    cols = ", ".join(values.keys())
    placeholders = ", ".join("?" for _ in values)
    cur = conn.execute(
        f"INSERT INTO orders ({cols}) VALUES ({placeholders})",
        tuple(values.values()),
    )
    return int(cur.lastrowid), True


def add_status_history(conn: sqlite3.Connection, order_id: int, status: str, status_at: str, source: str) -> None:
    conn.execute(
        "INSERT INTO order_status_history (order_id, status, status_at, source) "
        "VALUES (?, ?, ?, ?)",
        (order_id, status, status_at, source),
    )


def append_reject(rejects: list | None, source: str, file_name: str, row_index: int | None, reason: str, details: str) -> None:
    if rejects is None:
        return
    rejects.append(
        {
            "source": source,
            "file": file_name,
            "row_index": row_index,
            "reason": reason,
            "details": details,
        }
    )


def maybe_mark_delivered_from_payment(conn: sqlite3.Connection, sp_order_no: str) -> None:
    row = conn.execute(
        "SELECT id, status FROM orders WHERE sp_order_no = ?",
        (sp_order_no,),
    ).fetchone()
    if not row:
        return
    order_id, status = int(row[0]), str(row[1] or "")
    if status.lower() in {"poslato", "poslano"}:
        conn.execute(
            "UPDATE orders SET status = ? WHERE id = ?",
            ("Isporu\u010deno", order_id),
        )
        add_status_history(conn, order_id, "Isporu\u010deno", "", "SP-Uplate")

def import_sp_orders(conn: sqlite3.Connection, path: Path, rejects: list | None = None) -> None:
    df = pd.read_excel(path, sheet_name=SHEET_SP_ORDERS)
    import_id = start_import(conn, "SP-Narudzbe", path, len(df))
    if import_id is None:
        append_reject(rejects, "SP-Narudzbe", path.name, None, "file_already_imported", "")
        return

    seen_orders = set()
    max_sp_order_no = None
    for idx, row in df.iterrows():
        sp_order_no = str(row.get(COL["sp_order_no"], "")).strip()
        if not sp_order_no:
            continue
        try:
            sp_order_int = int(float(sp_order_no))
            if max_sp_order_no is None or sp_order_int > max_sp_order_no:
                max_sp_order_no = sp_order_int
        except ValueError:
            pass

        customer_key = compute_customer_key(
            row.get(COL["phone"], None),
            row.get(COL["email"], None),
            row.get(COL["customer_name"], None),
            row.get(COL["city"], None),
        )
        values = {
            "sp_order_no": sp_order_no,
            "woo_order_no": str(row.get(COL["woo_order_no"], "")).strip() or None,
            "client_code": str(row.get(COL["client"], "")).strip() or None,
            "tracking_code": str(row.get(COL["tracking"], "")).strip() or None,
            "customer_code": str(row.get(COL["customer_code"], "")).strip() or None,
            "customer_name": str(row.get(COL["customer_name"], "")).strip() or None,
            "city": str(row.get(COL["city"], "")).strip() or None,
            "address": str(row.get(COL["address"], "")).strip() or None,
            "postal_code": str(row.get(COL["postal_code"], "")).strip() or None,
            "phone": str(row.get(COL["phone"], "")).strip() or None,
            "email": str(row.get(COL["email"], "")).strip() or None,
            "customer_key": customer_key or None,
            "note": str(row.get(COL["note"], "")).strip() or None,
            "location": str(row.get(COL["location"], "")).strip() or None,
            "status": str(row.get(COL["status"], "")).strip() or None,
            "created_at": str(row.get(COL["created_at"], "")).strip() or None,
            "picked_up_at": str(row.get(COL["picked_up_at"], "")).strip() or None,
            "delivered_at": str(row.get(COL["delivered_at"], "")).strip() or None,
            "import_run_id": import_id,
        }

        order_id, created = get_or_create_order(conn, sp_order_no, values)
        if not created:
            append_reject(
                rejects,
                "SP-Narudzbe",
                path.name,
                int(idx) + 1,
                "order_exists",
                f"sp_order_no={sp_order_no}",
            )

        item_values = (
            order_id,
            str(row.get(COL["product_code"], "")).strip() or None,
            row.get(COL["qty"], None),
            row.get(COL["cod_amount"], None),
            row.get(COL["advance_amount"], None),
            row.get(COL["discount"], None),
            str(row.get(COL["discount_type"], "")).strip() or None,
            row.get(COL["addon_cod"], None),
            row.get(COL["addon_advance"], None),
            row.get(COL["extra_discount"], None),
            str(row.get(COL["extra_discount_type"], "")).strip() or None,
        )
        cur = conn.execute(
            "INSERT OR IGNORE INTO order_items ("
            "order_id, product_code, qty, cod_amount, advance_amount, "
            "discount, discount_type, addon_cod, addon_advance, "
            "extra_discount, extra_discount_type"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            item_values,
        )
        if cur.rowcount == 0:
            append_reject(
                rejects,
                "SP-Narudzbe",
                path.name,
                int(idx) + 1,
                "item_duplicate",
                f"sp_order_no={sp_order_no}, sku={values['product_code']}",
            )

        if sp_order_no not in seen_orders:
            status = values["status"] or ""
            status_at = values["delivered_at"] if status.lower() == "isporu\u010deno" else values["picked_up_at"]
            if status:
                add_status_history(conn, order_id, status, status_at or values["created_at"] or "", "SP-Narudzbe")
            seen_orders.add(sp_order_no)

    conn.commit()
    if max_sp_order_no is not None:
        set_app_state(conn, "last_sp_order_no", str(max_sp_order_no))
        conn.commit()


def import_sp_payments(conn: sqlite3.Connection, path: Path, rejects: list | None = None) -> None:
    df = pd.read_excel(path, sheet_name=SHEET_SP_PAYMENTS)
    import_id = start_import(conn, "SP-Uplate", path, len(df))
    if import_id is None:
        append_reject(rejects, "SP-Uplate", path.name, None, "file_already_imported", "")
        return

    for idx, row in df.iterrows():
        sp_order_no = str(row.get(COL["sp_order_no"], "")).strip()
        if not sp_order_no:
            continue
        values = (
            sp_order_no,
            str(row.get(COL["client"], "")).strip() or None,
            str(row.get(COL["customer_code"], "")).strip() or None,
            str(row.get(COL["payment_customer_name"], "")).strip() or None,
            row.get(COL["payment_amount"], None),
            str(row.get(COL["payment_order_status"], "")).strip() or None,
            str(row.get(COL["payment_client_status"], "")).strip() or None,
            import_id,
        )
        cur = conn.execute(
            "INSERT OR IGNORE INTO payments ("
            "sp_order_no, client_code, customer_code, customer_name, amount, "
            "order_status, client_status, import_run_id"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            values,
        )
        if cur.rowcount == 0:
            append_reject(
                rejects,
                "SP-Uplate",
                path.name,
                int(idx) + 1,
                "payment_duplicate",
                f"sp_order_no={sp_order_no}, amount={values[4]}, status={values[6]}",
            )
        maybe_mark_delivered_from_payment(conn, sp_order_no)

    conn.commit()


def import_sp_returns(conn: sqlite3.Connection, path: Path, rejects: list | None = None) -> None:
    df = pd.read_excel(path, sheet_name=SHEET_SP_ORDERS)
    import_id = start_import(conn, "SP-Preuzimanja", path, len(df))
    if import_id is None:
        append_reject(rejects, "SP-Preuzimanja", path.name, None, "file_already_imported", "")
        return

    for idx, row in df.iterrows():
        sp_order_no = str(row.get(COL["sp_order_no"], "")).strip() or None
        values = (
            sp_order_no,
            str(row.get(COL["tracking"], "")).strip() or None,
            str(row.get(COL["customer_name"], "")).strip() or None,
            str(row.get(COL["phone"], "")).strip() or None,
            str(row.get(COL["city"], "")).strip() or None,
            str(row.get(COL["status"], "")).strip() or None,
            str(row.get(COL["created_at"], "")).strip() or None,
            str(row.get(COL["picked_up_at"], "")).strip() or None,
            str(row.get(COL["delivered_at"], "")).strip() or None,
            import_id,
        )
        cur = conn.execute(
            "INSERT OR IGNORE INTO returns ("
            "sp_order_no, tracking_code, customer_name, phone, city, status, "
            "created_at, picked_up_at, delivered_at, import_run_id"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            values,
        )
        if cur.rowcount == 0:
            append_reject(
                rejects,
                "SP-Preuzimanja",
                path.name,
                int(idx) + 1,
                "return_duplicate",
                f"sp_order_no={sp_order_no}, tracking={values[1]}",
            )

    conn.commit()


def import_minimax(conn: sqlite3.Connection, path: Path, rejects: list | None = None) -> None:
    df = pd.read_excel(path, sheet_name=SHEET_MINIMAX)
    import_id = start_import(conn, "Minimax", path, len(df))
    if import_id is None:
        append_reject(rejects, "Minimax", path.name, None, "file_already_imported", "")
        return

    for idx, row in df.iterrows():
        values = (
            str(row.get(COL["mm_number"], "")).strip() or None,
            str(row.get(COL["mm_customer"], "")).strip() or None,
            str(row.get(COL["mm_country"], "")).strip() or None,
            str(row.get(COL["mm_date"], "")).strip() or None,
            str(row.get(COL["mm_due_date"], "")).strip() or None,
            str(row.get(COL["mm_revenue"], "")).strip() or None,
            row.get(COL["mm_amount_local"], None),
            row.get(COL["mm_amount_due"], None),
            str(row.get(COL["mm_analytics"], "")).strip() or None,
            str(row.get(COL["mm_turnover"], "")).strip() or None,
            str(row.get(COL["mm_account"], "")).strip() or None,
            str(row.get(COL["mm_basis"], "")).strip() or None,
            str(row.get(COL["mm_note"], "")).strip() or None,
            row.get(COL["mm_payment_amount"], None),
            row.get(COL["mm_open_amount"], None),
            import_id,
        )
        cur = conn.execute(
            "INSERT OR IGNORE INTO invoices ("
            "number, customer_name, country, date, due_date, revenue, "
            "amount_local, amount_due, analytics, turnover, account, basis, "
            "note, payment_amount, open_amount, import_run_id"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            values,
        )
        if cur.rowcount == 0:
            append_reject(
                rejects,
                "Minimax",
                path.name,
                int(idx) + 1,
                "invoice_duplicate",
                f"number={values[0]}",
            )

    conn.commit()
    apply_storno(conn)


def import_minimax_items(conn: sqlite3.Connection, path: Path) -> None:
    df = pd.read_excel(path, sheet_name=SHEET_MINIMAX)
    import_id = start_import(conn, "Minimax-Items", path, len(df))
    if import_id is None:
        return

    for _, row in df.iterrows():
        sku = str(row.get("Šifra", "")).strip().upper()
        if not sku:
            continue
        values = (
            sku,
            str(row.get("Naziv artikla", "")).strip() or None,
            str(row.get("Jedinica mere", "")).strip() or None,
            row.get("Masa(kg)", None),
            row.get("Stanje", None),
            row.get("Početna količina", None),
            row.get("Početna nabavna vrednost", None),
            row.get("Početna prodajna vrednost", None),
            row.get("Količina prijema", None),
            row.get("Nabavna vrednost prijema", None),
            row.get("Prodajna vrednost prijema", None),
            row.get("Količina izdavanja", None),
            row.get("Nabavna vrednost izdavanja", None),
            row.get("Prodajna vrednost izdavanja", None),
            row.get("Stanje.1", None),
            row.get("Konačna količina", None),
            row.get("Konačna nabavna vrednost", None),
            row.get("Konačna prodajna vrednost", None),
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            import_id,
        )
        conn.execute(
            "INSERT OR REPLACE INTO minimax_items ("
            "sku, name, unit, mass_kg, stock, "
            "opening_qty, opening_purchase_value, opening_sales_value, "
            "incoming_qty, incoming_purchase_value, incoming_sales_value, "
            "outgoing_qty, outgoing_purchase_value, outgoing_sales_value, "
            "stock_2, closing_qty, closing_purchase_value, closing_sales_value, "
            "updated_at, import_run_id"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            values,
        )

    conn.commit()

def normalize_date(value) -> date | None:
    if value is None or value == "":
        return None
    text = str(value)
    dayfirst = "." in text and "-" not in text
    ts = pd.to_datetime(value, errors="coerce", dayfirst=dayfirst)
    if pd.isna(ts):
        return None
    return ts.date()


def normalize_text(value: str | None) -> str:
    if not value:
        return ""
    return "".join(ch.lower() for ch in str(value) if ch.isalnum() or ch.isspace()).strip()

def normalize_text_loose(value: str | None) -> str:
    if not value:
        return ""
    text = str(value).lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return "".join(ch for ch in text if ch.isalnum() or ch.isspace()).strip()

def _levenshtein_leq_n(a: str, b: str, max_dist: int) -> bool:
    if a == b:
        return True
    if abs(len(a) - len(b)) > max_dist:
        return False
    if not a or not b:
        return False
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        min_row = i
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            val = min(
                prev[j] + 1,
                curr[j - 1] + 1,
                prev[j - 1] + cost,
            )
            curr.append(val)
            if val < min_row:
                min_row = val
        if min_row > max_dist:
            return False
        prev = curr
    return prev[-1] <= max_dist

def fuzzy_contains(text: str, pattern: str, max_dist: int = 3) -> bool:
    if not text or not pattern:
        return False
    if pattern in text:
        return True
    t_len = len(text)
    p_len = len(pattern)
    min_len = max(1, p_len - max_dist)
    max_len = p_len + max_dist
    for win_len in range(min_len, max_len + 1):
        if win_len > t_len:
            break
        for i in range(0, t_len - win_len + 1):
            chunk = text[i : i + win_len]
            if _levenshtein_leq_n(chunk, pattern, max_dist):
                return True
    return False

def extract_invoice_no_from_text(text: str | None) -> tuple[str | None, str | None]:
    if not text:
        return None, None
    match = re.search(r"\bSP-MM-\d+\b", text, flags=re.I)
    if match:
        val = match.group(0)
        digits = re.sub(r"\D", "", val)
        return val, digits or None
    match = re.search(r"\b\d{8,12}\b", text)
    if match:
        val = match.group(0)
        return val, val
    return None, None

def classify_refund_reason(purpose: str | None) -> str | None:
    text = normalize_text_loose(purpose)
    if not text:
        return None
    patterns = [
        ("reklamirana roba povrat sredstava", "reklamirana_roba_povrat_sredstava"),
        ("reklamacija robe povrat sredstava", "reklamacija_robe_povrat_sredstava"),
        ("povrat kupljene robe povrat sredstava", "povrat_kupljene_robe_povrat_sredstava"),
        ("povrat robe storno racuna", "povrat_robe_storno_racuna"),
        ("povrat robe storno", "povrat_robe_storno"),
        ("storno racuna", "storno_racuna"),
        ("povrat robe", "povrat_robe"),
    ]
    for pattern, reason in patterns:
        if fuzzy_contains(text, pattern, max_dist=3):
            return reason
    return None

def invoice_digits(value: str | None) -> str:
    if not value:
        return ""
    return re.sub(r"\D", "", str(value))


def amount_exact_strict(a, b) -> bool:
    if a is None or b is None:
        return False
    try:
        return round(float(a), 2) == round(float(b), 2)
    except (TypeError, ValueError):
        return False


def name_exact_strict(a: str | None, b: str | None) -> bool:
    return normalize_text(a) == normalize_text(b)


def _levenshtein_leq_one(a: str, b: str) -> bool:
    if a == b:
        return True
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        mismatches = 0
        for ch1, ch2 in zip(a, b):
            if ch1 != ch2:
                mismatches += 1
                if mismatches > 1:
                    return False
        return True
    if len(a) > len(b):
        a, b = b, a
    i = 0
    j = 0
    skips = 0
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            i += 1
            j += 1
        else:
            skips += 1
            if skips > 1:
                return False
            j += 1
    return True


def name_distance_ok(a: str | None, b: str | None, max_distance: int = 1) -> bool:
    if max_distance != 1:
        raise ValueError("Only max_distance=1 is supported.")
    a_norm = normalize_text(a)
    b_norm = normalize_text(b)
    if not a_norm or not b_norm:
        return False
    a_parts = [p for p in a_norm.split() if p]
    b_parts = [p for p in b_norm.split() if p]
    if not a_parts or not b_parts:
        return False
    a_first = a_parts[0]
    a_last = a_parts[-1] if len(a_parts) > 1 else a_parts[0]
    b_first = b_parts[0]
    b_last = b_parts[-1] if len(b_parts) > 1 else b_parts[0]
    return _levenshtein_leq_one(a_first, b_first) and _levenshtein_leq_one(a_last, b_last)


def is_cancelled_status(status: str | None) -> bool:
    text = normalize_text(status)
    return "otkazan" in text


def is_in_progress_status(status: str | None) -> bool:
    text = normalize_text(status)
    return "u obradi" in text


def is_unpicked_status(status: str | None) -> bool:
    text = normalize_text_loose(status)
    return text.startswith("vrac")


def normalize_phone(value: str | None) -> str:
    if not value:
        return ""
    digits = "".join(ch for ch in str(value) if ch.isdigit())
    if not digits:
        return ""
    if digits.startswith("3810"):
        digits = "0" + digits[4:]
    elif digits.startswith("381"):
        digits = "0" + digits[3:]
    if digits.startswith("0"):
        return digits
    if digits.startswith("6") and 8 <= len(digits) <= 10:
        return "0" + digits
    return digits


def compute_customer_key(phone: str | None, email: str | None, name: str | None, city: str | None) -> str:
    phone_key = normalize_phone(phone)
    if not phone_key and email:
        email_text = str(email)
        if "@" not in email_text:
            phone_key = normalize_phone(email_text)
    if phone_key:
        return f"phone:{phone_key}"
    email_key = normalize_text(email) if email and "@" in str(email) else ""
    if email_key:
        return f"email:{email_key}"
    name_key = normalize_text(name)
    city_key = normalize_text(city)
    if name_key or city_key:
        return f"name:{name_key}|city:{city_key}"
    return ""


def parse_tracking_time(value: str | None) -> str | None:
    if not value:
        return None
    ts = pd.to_datetime(value, errors="coerce", dayfirst=True)
    if pd.isna(ts):
        return None
    return ts.strftime("%Y-%m-%d %H:%M:%S")


def _status_has(status: str | None, needle: str) -> bool:
    if not status or not needle:
        return False
    return needle in normalize_text_loose(status)


def analyze_tracking_history(history: list[dict]) -> tuple[list[tuple[str | None, str | None]], dict]:
    events = []
    for entry in history:
        status_time = parse_tracking_time(entry.get("statusTime"))
        status_value = entry.get("statusValue")
        events.append((status_time, status_value))
    events_sorted = sorted(
        events,
        key=lambda x: (x[0] or "9999-12-31 23:59:59"),
    )

    received_at = None
    first_out_for_delivery_at = None
    returned_at = None
    delivery_attempts = 0
    reasons = []
    last_status = None
    last_status_at = None
    for status_time, status_value in events_sorted:
        text = normalize_text_loose(status_value)
        if status_value:
            last_status = status_value
            last_status_at = status_time
        if not received_at and _status_has(status_value, "preuzeta od posiljaoca"):
            received_at = status_time
        if _status_has(status_value, "zaduzena za isporuku"):
            delivery_attempts += 1
            if not first_out_for_delivery_at:
                first_out_for_delivery_at = status_time
        if "pokusaj isporuke" in text or "ponovni pokusaj" in text:
            delivery_attempts += 1
        if _status_has(status_value, "vracena posiljaocu") or _status_has(status_value, "vraca se posiljaocu"):
            returned_at = status_time
        if any(
            marker in text
            for marker in [
                "telefon",
                "netacan",
                "nema nikoga",
                "nema nikog",
                "nema na adresi",
                "odbij",
                "odbio",
                "nepoznat",
                "pogresna adresa",
                "adresa",
                "neuspes",
                "bezuspes",
                "nije dostup",
                "ne moze",
                "nepostojec",
                "neispravan",
            ]
        ):
            if status_value and status_value not in reasons:
                reasons.append(status_value)

    days_to_first_attempt = None
    if received_at and first_out_for_delivery_at:
        try:
            dt_received = pd.to_datetime(received_at)
            dt_first = pd.to_datetime(first_out_for_delivery_at)
            days_to_first_attempt = (dt_first - dt_received).total_seconds() / 86400.0
        except Exception:
            days_to_first_attempt = None

    has_attempt_before_return = 0
    if returned_at:
        try:
            dt_returned = pd.to_datetime(returned_at)
            for status_time, status_value in events_sorted:
                if not status_time:
                    continue
                if _status_has(status_value, "zaduzena za isporuku") or _status_has(status_value, "pokusaj isporuke"):
                    dt = pd.to_datetime(status_time)
                    if dt <= dt_returned:
                        has_attempt_before_return = 1
                        break
        except Exception:
            has_attempt_before_return = 0

    anomalies = []
    text_reasons = normalize_text_loose(" ".join(reasons))
    if "telefon" in text_reasons and "nema nikoga" in text_reasons:
        anomalies.append("Nelogican slijed: telefon netacan + nema nikoga")
    if returned_at and not has_attempt_before_return:
        anomalies.append("Vracena bez pokusaja isporuke")
    if not returned_at:
        anomalies.append("Nema statusa vracena posiljaocu")

    summary = {
        "received_at": received_at,
        "first_out_for_delivery_at": first_out_for_delivery_at,
        "delivery_attempts": delivery_attempts,
        "failure_reasons": "; ".join(reasons),
        "returned_at": returned_at,
        "days_to_first_attempt": days_to_first_attempt,
        "has_attempt_before_return": has_attempt_before_return,
        "has_returned": 1 if returned_at else 0,
        "anomalies": "; ".join(anomalies),
        "last_status": last_status,
        "last_status_at": last_status_at,
    }
    return events_sorted, summary


def fetch_dexpress_tracking(tracking_code: str) -> tuple[dict | None, int | None]:
    if not tracking_code:
        return None, None
    import urllib.parse

    url = "https://www.dexpress.rs/rs/pracenje-posiljaka"
    form_data = {
        "ajax": "yes",
        "task": "search",
        "data[package_tracking_search]": tracking_code,
    }
    encoded = urllib.parse.urlencode(form_data).encode("utf-8")
    headers = {
        "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
        "X-Requested-With": "XMLHttpRequest",
        "Origin": "https://www.dexpress.rs",
        "Referer": f"https://www.dexpress.rs/rs/pracenje-posiljaka/{tracking_code}",
    }
    with http_request(url, method="POST", data=encoded, headers=headers, timeout=20) as resp:
        status_code = getattr(resp, "status", None)
        payload = resp.read().decode("utf-8", errors="ignore")
    try:
        return json.loads(payload), status_code
    except json.JSONDecodeError:
        return None, status_code


def fetch_slanjepaketa_tracking(tracking_code: str) -> tuple[dict | None, int | None]:
    if not tracking_code:
        return None, None
    url = f"https://softver.slanjepaketa.rs/api/v1/product-orders/status-info/{tracking_code}"
    headers = {
        "Authorization": "External 7b028ded-ebe4-4d74-ae79-ff516a64a851",
        "Referer": f"https://www.slanjepaketa.rs/pracenje-posiljaka/{tracking_code}",
        "Origin": "https://www.slanjepaketa.rs",
    }
    with http_request(url, method="GET", headers=headers, timeout=20) as resp:
        status_code = getattr(resp, "status", None)
        payload = resp.read().decode("utf-8", errors="ignore")
    try:
        return json.loads(payload), status_code
    except json.JSONDecodeError:
        return None, status_code


def tracking_public_url(tracking_code: str) -> str:
    if tracking_code.upper().startswith("SPF"):
        return f"https://www.slanjepaketa.rs/pracenje-posiljaka/{tracking_code}"
    return f"https://www.dexpress.rs/rs/pracenje-posiljaka/{tracking_code}"


def save_tracking_result(
    conn: sqlite3.Connection,
    tracking_code: str,
    events: list[tuple[str | None, str | None]],
    summary: dict,
) -> None:
    fetched_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn.executemany(
        "INSERT OR IGNORE INTO tracking_events (tracking_code, status_time, status_value, fetched_at) "
        "VALUES (?, ?, ?, ?)",
        [(tracking_code, t, v, fetched_at) for t, v in events],
    )
    conn.execute(
        "INSERT INTO tracking_summary "
        "(tracking_code, received_at, first_out_for_delivery_at, delivery_attempts, "
        "failure_reasons, returned_at, days_to_first_attempt, has_attempt_before_return, "
        "has_returned, anomalies, last_status, last_status_at, last_fetched_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
        "ON CONFLICT(tracking_code) DO UPDATE SET "
        "received_at=excluded.received_at, "
        "first_out_for_delivery_at=excluded.first_out_for_delivery_at, "
        "delivery_attempts=excluded.delivery_attempts, "
        "failure_reasons=excluded.failure_reasons, "
        "returned_at=excluded.returned_at, "
        "days_to_first_attempt=excluded.days_to_first_attempt, "
        "has_attempt_before_return=excluded.has_attempt_before_return, "
        "has_returned=excluded.has_returned, "
        "anomalies=excluded.anomalies, "
        "last_status=excluded.last_status, "
        "last_status_at=excluded.last_status_at, "
        "last_fetched_at=excluded.last_fetched_at",
        (
            tracking_code,
            summary.get("received_at"),
            summary.get("first_out_for_delivery_at"),
            summary.get("delivery_attempts"),
            summary.get("failure_reasons"),
            summary.get("returned_at"),
            summary.get("days_to_first_attempt"),
            summary.get("has_attempt_before_return"),
            summary.get("has_returned"),
            summary.get("anomalies"),
            summary.get("last_status"),
            summary.get("last_status_at"),
            fetched_at,
        ),
    )
    conn.commit()


def log_tracking_request(
    tracking_code: str,
    url: str,
    status_code: int | None,
    latency_ms: int | None,
    result: str,
    error: str | None = None,
) -> None:
    log_dir = Path("exports")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "tracking-log.csv"
    file_exists = log_path.exists()
    with log_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if not file_exists:
            writer.writerow(
                [
                    "timestamp",
                    "tracking_code",
                    "url",
                    "status_code",
                    "latency_ms",
                    "result",
                    "error",
                ]
            )
        writer.writerow(
            [
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                tracking_code,
                url,
                status_code if status_code is not None else "",
                latency_ms if latency_ms is not None else "",
                result,
                error or "",
            ]
        )


def log_app_error(source: str, message: str) -> None:
    log_dir = Path("exports")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "app-errors.log"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{timestamp}] {source}: {message}\n")


def get_latest_task_progress(conn: sqlite3.Connection) -> dict | None:
    row = conn.execute(
        "SELECT task, total, processed, updated_at "
        "FROM task_progress "
        "ORDER BY updated_at DESC LIMIT 1"
    ).fetchone()
    if not row:
        return None
    return {
        "task": row[0],
        "total": int(row[1]),
        "processed": int(row[2]),
        "updated_at": row[3],
    }


def load_review_samples(conn: sqlite3.Connection, limit: int = 30) -> list[dict]:
    rows = conn.execute(
        "SELECT m.id, m.score, o.id, o.sp_order_no, o.customer_name, o.picked_up_at, "
        "i.id, i.number, i.customer_name, i.date, i.amount_due "
        "FROM invoice_matches m "
        "JOIN orders o ON o.id = m.order_id "
        "JOIN invoices i ON i.id = m.invoice_id "
        "WHERE m.status = 'review' "
        "ORDER BY m.score DESC "
        "LIMIT ?",
        (limit,),
    ).fetchall()
    if not rows:
        return []
    order_ids = [int(r[2]) for r in rows]
    totals = build_order_net_map(conn, order_ids)
    result = []
    for (
        match_id,
        score,
        order_id,
        sp_order_no,
        order_name,
        order_date,
        invoice_id,
        invoice_no,
        invoice_name,
        invoice_date,
        amount_due,
    ) in rows:
        result.append(
            {
                "match_id": match_id,
                "score": score,
                "order_id": order_id,
                "sp_order_no": sp_order_no,
                "order_name": order_name,
                "order_date": order_date,
                "order_amount": totals.get(int(order_id), 0.0),
                "invoice_id": invoice_id,
                "invoice_no": invoice_no,
                "invoice_name": invoice_name,
                "invoice_date": invoice_date,
                "invoice_amount": float(amount_due or 0),
            }
        )
    return result


def load_all_match_samples(conn: sqlite3.Connection, limit: int = 30) -> list[dict]:
    rows = conn.execute(
        "SELECT m.id, m.score, o.id, o.sp_order_no, o.customer_name, o.picked_up_at, "
        "i.id, i.number, i.customer_name, i.date, i.amount_due "
        "FROM invoice_matches m "
        "JOIN orders o ON o.id = m.order_id "
        "JOIN invoices i ON i.id = m.invoice_id "
        "ORDER BY m.score DESC "
        "LIMIT ?",
        (limit,),
    ).fetchall()
    if not rows:
        return []
    order_ids = [int(r[2]) for r in rows]
    totals = build_order_net_map(conn, order_ids)
    result = []
    for (
        match_id,
        score,
        order_id,
        sp_order_no,
        order_name,
        order_date,
        invoice_id,
        invoice_no,
        invoice_name,
        invoice_date,
        amount_due,
    ) in rows:
        result.append(
            {
                "match_id": match_id,
                "score": score,
                "order_id": order_id,
                "sp_order_no": sp_order_no,
                "order_name": order_name,
                "order_date": order_date,
                "order_amount": totals.get(int(order_id), 0.0),
                "invoice_id": invoice_id,
                "invoice_no": invoice_no,
                "invoice_name": invoice_name,
                "invoice_date": invoice_date,
                "invoice_amount": float(amount_due or 0),
            }
        )
    return result


def _should_skip_tracking(
    conn: sqlite3.Connection,
    tracking_code: str,
    fast_hours: int = 6,
    slow_hours: int = 24,
) -> bool:
    row = conn.execute(
        "SELECT last_fetched_at, has_returned, delivery_attempts, last_status "
        "FROM tracking_summary WHERE tracking_code = ?",
        (tracking_code,),
    ).fetchone()
    if not row:
        return False
    last_fetched_at, has_returned, delivery_attempts, last_status = row
    if has_returned:
        return True
    if last_status and _status_has(last_status, "isporucena"):
        return True
    if not last_fetched_at:
        return False
    try:
        last_dt = pd.to_datetime(last_fetched_at)
    except Exception:
        return False
    if pd.isna(last_dt):
        return False
    delta_hours = (datetime.now() - last_dt).total_seconds() / 3600.0
    if delivery_attempts and delivery_attempts > 0:
        return delta_hours < fast_hours
    return delta_hours < slow_hours


def update_unpicked_tracking(
    conn: sqlite3.Connection,
    batch_size: int = 20,
    min_delay: int = 8,
    max_delay: int = 12,
    batch_pause_min: int = 180,
    batch_pause_max: int = 300,
    backoff_min: int = 120,
    backoff_max: int = 600,
    progress_task: str | None = None,
    force_refresh: bool = False,
) -> int:
    import random
    import time
    import urllib.error

    rows = _unpicked_rows(conn)
    tracking_codes = sorted(
        {str(r[9]).strip() for r in rows if len(r) > 9 and r[9]}
    )
    if not tracking_codes:
        return 0
    total = len(tracking_codes)
    scanned = 0
    if progress_task:
        set_task_progress(conn, progress_task, total)
        update_task_progress(conn, progress_task, 0)
    processed = 0
    for idx, code in enumerate(tracking_codes, start=1):
        url = tracking_public_url(code)
        if not force_refresh and _should_skip_tracking(conn, code):
            log_tracking_request(code, url, None, None, "skipped_cache", None)
            scanned += 1
            if progress_task:
                update_task_progress(conn, progress_task, scanned)
            continue
        attempts = 0
        max_retries = 2
        while True:
            start_time = time.time()
            try:
                if code.upper().startswith("SPF"):
                    data, status_code = fetch_slanjepaketa_tracking(code)
                    if not data or "notes" not in data:
                        latency_ms = int((time.time() - start_time) * 1000)
                        log_tracking_request(code, url, status_code, latency_ms, "no_data", None)
                        break
                    history = [
                        {"statusTime": note.get("date"), "statusValue": note.get("note")}
                        for note in (data.get("notes") or [])
                    ]
                else:
                    data, status_code = fetch_dexpress_tracking(code)
                    if not data or not data.get("flag"):
                        latency_ms = int((time.time() - start_time) * 1000)
                        log_tracking_request(code, url, status_code, latency_ms, "no_data", None)
                        break
                    history = data.get("historyStatuses") or []
                events, summary = analyze_tracking_history(history)
                save_tracking_result(conn, code, events, summary)
                processed += 1
                latency_ms = int((time.time() - start_time) * 1000)
                log_tracking_request(code, url, status_code, latency_ms, "ok", None)
                break
            except urllib.error.HTTPError as exc:
                latency_ms = int((time.time() - start_time) * 1000)
                log_tracking_request(code, url, exc.code, latency_ms, "http_error", str(exc))
                if exc.code in (403, 429):
                    time.sleep(random.uniform(backoff_min, backoff_max))
                    attempts += 1
                    if attempts <= max_retries:
                        continue
                    break
                raise
            except urllib.error.URLError as exc:
                latency_ms = int((time.time() - start_time) * 1000)
                log_tracking_request(code, url, None, latency_ms, "timeout", str(exc))
                time.sleep(random.uniform(backoff_min, backoff_max))
                attempts += 1
                if attempts <= max_retries:
                    continue
                break
            except Exception as exc:
                latency_ms = int((time.time() - start_time) * 1000)
                log_tracking_request(code, url, None, latency_ms, "error", str(exc))
                break

        scanned += 1
        if progress_task:
            update_task_progress(conn, progress_task, scanned)

        if idx % batch_size == 0:
            time.sleep(random.uniform(batch_pause_min, batch_pause_max))
        else:
            time.sleep(random.uniform(min_delay, max_delay))
    if progress_task:
        update_task_progress(conn, progress_task, total)
    return processed

def recompute_customer_keys(conn: sqlite3.Connection) -> int:
    rows = conn.execute(
        "SELECT id, phone, email, customer_name, city FROM orders"
    ).fetchall()
    updates = []
    for order_id, phone, email, name, city in rows:
        key = compute_customer_key(phone, email, name, city) or None
        updates.append((key, order_id))
    if updates:
        conn.executemany("UPDATE orders SET customer_key = ? WHERE id = ?", updates)
        conn.commit()
    return len(updates)


def ensure_customer_keys(conn: sqlite3.Connection) -> int:
    version = get_app_state(conn, "customer_key_version")
    if version == CUSTOMER_KEY_VERSION:
        return 0
    updated = recompute_customer_keys(conn)
    set_app_state(conn, "customer_key_version", CUSTOMER_KEY_VERSION)
    return updated


def extract_sp_order_no(note: str) -> str | None:
    if not note:
        return None
    match = re.search(r"\b(\d{4,})\b", note)
    if not match:
        return None
    return match.group(1)


def compute_order_component(max_val, min_val, sum_val) -> float | None:
    if max_val is None:
        return None
    if min_val is None:
        return float(max_val)
    try:
        max_f = float(max_val)
        min_f = float(min_val)
        sum_f = float(sum_val) if sum_val is not None else None
        if sum_f is not None and sum_f > max_f + 0.01:
            return sum_f
        if abs(max_f - min_f) < 0.01:
            return max_f
        return sum_f if sum_f is not None else max_f
    except (TypeError, ValueError):
        return None


def compute_order_amount(cod, addon, advance, addon_advance) -> float | None:
    if cod is None and addon is None and advance is None and addon_advance is None:
        return None
    base = float(cod or 0) + float(addon or 0)
    paid = float(advance or 0) + float(addon_advance or 0)
    return base - paid


def apply_percent(value, percent) -> float:
    val = to_float(value) or 0.0
    pct = to_float(percent)
    if pct is None:
        return val
    if pct < 0 or pct > 100:
        return val
    return val * (1 - pct / 100.0)


def build_order_net_map(conn: sqlite3.Connection, order_ids: list[int]) -> dict[int, float]:
    if not order_ids:
        return {}
    net_map: dict[int, float] = {int(oid): 0.0 for oid in order_ids}
    chunk_size = 900
    for i in range(0, len(order_ids), chunk_size):
        chunk = order_ids[i : i + chunk_size]
        placeholders = ",".join("?" for _ in chunk)
        rows = conn.execute(
            "SELECT order_id, cod_amount, addon_cod, advance_amount, addon_advance, "
            "discount, extra_discount "
            f"FROM order_items WHERE order_id IN ({placeholders})",
            chunk,
        ).fetchall()
        for row in rows:
            order_id = int(row[0])
            cod = apply_percent(row[1], row[5])
            addon = apply_percent(row[2], row[6])
            advance = to_float(row[3]) or 0.0
            addon_advance = to_float(row[4]) or 0.0
            net_map[order_id] = net_map.get(order_id, 0.0) + cod + addon - advance - addon_advance
    return net_map


def is_no_value_order(cod, addon, advance, addon_advance) -> bool:
    vals = [cod, addon, advance, addon_advance]
    total = 0.0
    for val in vals:
        if val is None:
            continue
        try:
            total += abs(float(val))
        except (TypeError, ValueError):
            continue
    return total == 0.0


def score_match(order, invoice) -> int:
    score, _ = score_match_with_reasons(order, invoice)
    return score


def score_match_with_reasons(order, invoice) -> tuple[int, list[str]]:
    score = 0
    reasons = []
    order_date = normalize_date(order["picked_up_at"])
    inv_date = normalize_date(invoice["turnover"])
    if order_date and inv_date:
        delta = (inv_date - order_date).days
        if -10 <= delta <= 10:
            score += 30
            reasons.append("date-10+10")
    order_amount = order["amount"]
    inv_amount = invoice["amount_due"]
    if amount_exact_strict(order_amount, inv_amount):
        score += 40
        reasons.append("amount_exact")
    order_name = order.get("customer_name")
    invoice_name = invoice.get("customer_name")
    if name_exact_strict(order_name, invoice_name):
        score += 30
        reasons.append("name_exact")
    elif name_distance_ok(order_name, invoice_name):
        score += 30
        reasons.append("name_close")
    return score, reasons


def extract_invoice_number_from_basis(basis: str | None) -> str | None:
    if not basis:
        return None
    match = re.search(r"\b([A-Z]{2}-[A-Z]{2}-\d+)\b", basis)
    if not match:
        return None
    return match.group(1)


def extract_invoice_number_from_text(text: str | None) -> str | None:
    if not text:
        return None
    match = re.search(r"\bSP-MM-\d+\b", text)
    if not match:
        return None
    return match.group(0)


def to_float(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def get_text(elem, path: str) -> str:
    if elem is None:
        return ""
    child = elem.find(path)
    if child is None or child.text is None:
        return ""
    return child.text.strip()


def import_bank_xml(conn: sqlite3.Connection, path: Path, rejects: list | None = None) -> None:
    tree = ET.parse(path)
    root = tree.getroot()

    rows = []
    for stmtrs in root.findall(".//stmtrs"):
        stmt_number = get_text(stmtrs, "stmtnumber")
        for trn in stmtrs.findall(".//stmttrn"):
            rows.append(
                (
                    get_text(trn, "fitid"),
                    stmt_number,
                    get_text(trn, "benefit"),
                    get_text(trn, "dtposted"),
                    to_float(get_text(trn, "trnamt")),
                    get_text(trn, "purpose"),
                    get_text(trn, "purposecode"),
                    get_text(trn, "payeeinfo/name"),
                    get_text(trn, "payeeinfo/city"),
                    get_text(trn, "payeeaccountinfo/acctid"),
                    get_text(trn, "payeeaccountinfo/bankid"),
                    get_text(trn, "payeeaccountinfo/bankname"),
                    get_text(trn, "refnumber"),
                    get_text(trn, "payeerefnumber"),
                    get_text(trn, "urgency"),
                    to_float(get_text(trn, "fee")),
                )
            )

    import_id = start_import(conn, "Bank-XML", path, len(rows))
    if import_id is None:
        append_reject(rejects, "Bank-XML", path.name, None, "file_already_imported", "")
        return

    for idx, row in enumerate(rows):
        cur = conn.execute(
            "INSERT OR IGNORE INTO bank_transactions ("
            "fitid, stmt_number, benefit, dtposted, amount, purpose, purposecode, "
            "payee_name, payee_city, payee_acctid, payee_bankid, payee_bankname, "
            "refnumber, payeerefnumber, urgency, fee, import_run_id"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (*row, import_id),
        )
        if cur.rowcount == 0:
            append_reject(
                rejects,
                "Bank-XML",
                path.name,
                int(idx) + 1,
                "bank_txn_duplicate",
                f"fitid={row[0]}",
            )
    conn.commit()


def apply_storno(conn: sqlite3.Connection) -> None:
    rows = conn.execute(
        "SELECT id, amount_due, revenue, basis FROM invoices"
    ).fetchall()
    for row in rows:
        _, amount_due, revenue, basis = row
        amount_due_val = to_float(amount_due)
        revenue_val = to_float(revenue)
        is_storno = (amount_due_val is not None and amount_due_val < 0) or (
            revenue_val is not None and revenue_val < 0
        )
        if not is_storno:
            continue
        basis_no = extract_invoice_number_from_basis(str(basis or ""))
        if not basis_no:
            continue
        orig = conn.execute(
            "SELECT id, amount_due FROM invoices WHERE number = ?",
            (basis_no,),
        ).fetchone()
        if not orig:
            continue
        orig_id, orig_amount = int(orig[0]), to_float(orig[1])
        if orig_amount is None:
            continue
        storno_val = abs(amount_due_val or revenue_val or 0)
        if abs(orig_amount - storno_val) <= 2.0:
            new_open = 0.0
        else:
            new_open = max(0.0, orig_amount - storno_val)
        conn.execute(
            "UPDATE invoices SET open_amount = ? WHERE id = ?",
            (new_open, orig_id),
        )
        conn.execute(
            "INSERT OR REPLACE INTO invoice_storno ("
            "storno_invoice_id, original_invoice_id, storno_amount, "
            "original_amount, remaining_open, is_partial, created_at"
            ") VALUES (?, ?, ?, ?, ?, ?, datetime('now'))",
            (
                row[0],
                orig_id,
                storno_val,
                orig_amount,
                new_open,
                1 if new_open > 0 else 0,
            ),
        )
    conn.commit()

def match_minimax(
    conn: sqlite3.Connection,
    auto_threshold: int = 70,
    review_threshold: int = 50,
    progress_task: str | None = None,
) -> None:
    conn.execute("DELETE FROM order_flags WHERE flag = 'needs_invoice'")
    orders = []
    rows = conn.execute(
        "SELECT o.id, o.sp_order_no, o.customer_name, o.picked_up_at, o.created_at, "
        "o.phone, o.address, o.city, o.status, "
        "MAX(oi.cod_amount), MIN(oi.cod_amount), SUM(oi.cod_amount), "
        "MAX(oi.addon_cod), MIN(oi.addon_cod), SUM(oi.addon_cod), "
        "MAX(oi.advance_amount), MIN(oi.advance_amount), SUM(oi.advance_amount), "
        "MAX(oi.addon_advance), MIN(oi.addon_advance), SUM(oi.addon_advance), "
        "MAX(oi.discount), MIN(oi.discount), COUNT(oi.id) "
        "FROM orders o LEFT JOIN order_items oi ON oi.order_id = o.id "
        "WHERE o.id NOT IN (SELECT order_id FROM invoice_matches) "
        "GROUP BY o.id"
    ).fetchall()
    order_ids = [int(row[0]) for row in rows]
    net_map = build_order_net_map(conn, order_ids)
    def is_all_zero_values(max_cod, max_addon, max_adv, max_addon_adv, item_count: int) -> bool:
        if item_count == 0:
            return True
        vals = [max_cod, max_addon, max_adv, max_addon_adv]
        for val in vals:
            try:
                if val is not None and abs(float(val)) > 0.0:
                    return False
            except (TypeError, ValueError):
                continue
        return True

    def is_all_discount_100(min_discount, max_discount, item_count: int) -> bool:
        if item_count == 0:
            return False
        try:
            return float(min_discount) == 100.0 and float(max_discount) == 100.0
        except (TypeError, ValueError):
            return False

    for row in rows:
        status = row[8]
        if is_cancelled_status(status) or is_in_progress_status(status):
            continue
        order_id = int(row[0])
        order_date = normalize_date(row[3] or row[4])
        amount = net_map.get(order_id)
        max_cod = row[9]
        max_addon = row[12]
        max_adv = row[15]
        max_addon_adv = row[18]
        max_discount = row[21]
        min_discount = row[22]
        item_count = int(row[23] or 0)
        if is_all_zero_values(max_cod, max_addon, max_adv, max_addon_adv, item_count):
            continue
        if is_all_discount_100(min_discount, max_discount, item_count):
            continue
        if amount is None:
            continue
        orders.append(
            {
                "id": order_id,
                "sp_order_no": str(row[1]),
                "customer_name": row[2],
                "picked_up_at": row[3] or row[4],
                "phone": row[5],
                "address": row[6],
                "city": row[7],
                "amount": amount,
                "picked_up_date": order_date,
            }
        )

    invoices = []
    inv_rows = conn.execute(
        "SELECT id, number, customer_name, turnover, amount_due, note, analytics, account "
        "FROM invoices "
        "WHERE id NOT IN (SELECT invoice_id FROM invoice_matches)"
    ).fetchall()
    for row in inv_rows:
        invoices.append(
            {
                "id": int(row[0]),
                "number": row[1],
                "customer_name": row[2],
                "turnover": row[3],
                "amount_due": row[4],
                "note": row[5],
                "analytics": row[6],
                "account": row[7],
                "turnover_date": normalize_date(row[3]),
            }
        )

    total_steps = len(invoices) + len(orders) * 2 + 2
    processed_steps = 0
    last_progress = 0
    progress_every = 10
    if progress_task:
        set_task_progress(conn, progress_task, total_steps)

    def maybe_update_progress():
        nonlocal last_progress
        if not progress_task:
            return
        if processed_steps - last_progress >= progress_every or processed_steps >= total_steps:
            update_task_progress(conn, progress_task, processed_steps)
            last_progress = processed_steps

    processed_steps += 1
    maybe_update_progress()

    def amount_exact(a, b) -> bool:
        return amount_exact_strict(a, b)

    def name_exact(a, b) -> bool:
        return name_exact_strict(a, b)

    def name_close(a, b) -> bool:
        return name_distance_ok(a, b, max_distance=1)

    def date_in_window(d1, d2, days_back: int = 10, days_forward: int = 10) -> bool:
        if not d1 or not d2:
            return False
        delta = (d2 - d1).days
        return -days_back <= delta <= days_forward

    orders_by_date = {}
    orders_no_date = []
    for order in orders:
        od = order.get("picked_up_date")
        if od:
            orders_by_date.setdefault(od, []).append(order)
        else:
            orders_no_date.append(order)

    invoices_by_date = {}
    invoices_no_date = []
    for inv in invoices:
        idate = inv.get("turnover_date")
        if idate:
            invoices_by_date.setdefault(idate, []).append(inv)
        else:
            invoices_no_date.append(inv)

    def candidate_orders_for_invoice(inv):
        idate = inv.get("turnover_date")
        if idate:
            date_candidates = []
            for offset in range(-10, 11):
                date_candidates.extend(
                    orders_by_date.get(idate + timedelta(days=offset), [])
                )
            date_candidates.extend(orders_no_date)
        else:
            date_candidates = orders
        filtered = [
            o for o in date_candidates
            if amount_exact(o.get("amount"), inv.get("amount_due"))
            and date_in_window(o.get("picked_up_date"), idate, 7)
        ]
        if filtered:
            return filtered
        return date_candidates

    def candidate_invoices_for_order(order, pool):
        odate = order.get("picked_up_date")
        if odate:
            date_candidates = []
            for offset in range(-10, 11):
                date_candidates.extend(
                    invoices_by_date.get(odate + timedelta(days=offset), [])
                )
            date_candidates.extend(invoices_no_date)
        else:
            date_candidates = pool
        filtered = [
            inv for inv in date_candidates
            if amount_exact(order.get("amount"), inv.get("amount_due"))
            and date_in_window(odate, inv.get("turnover_date"), 7)
        ]
        if filtered:
            return filtered
        return date_candidates

    used_orders = set()
    matched_invoice_ids = {
        int(row[0]) for row in conn.execute("SELECT invoice_id FROM invoice_matches").fetchall()
    }

    def select_best_invoice(order, candidates):
        odate = order.get("picked_up_date")
        best = None
        best_key = None
        for inv in candidates:
            idate = inv.get("turnover_date")
            if odate and idate:
                delta = abs((idate - odate).days)
            else:
                delta = 9999
            key = (delta, inv["id"])
            if best_key is None or key < best_key:
                best_key = key
                best = inv
        return best

    def find_candidates(order, name_check):
        odate = order.get("picked_up_date")
        if odate:
            date_candidates = []
            for offset in range(-10, 11):
                date_candidates.extend(
                    invoices_by_date.get(odate + timedelta(days=offset), [])
                )
            date_candidates.extend(invoices_no_date)
        else:
            date_candidates = invoices
        candidates = [
            inv for inv in date_candidates
            if inv["id"] not in matched_invoice_ids
            and name_check(order.get("customer_name"), inv.get("customer_name"))
            and amount_exact(order.get("amount"), inv.get("amount_due"))
            and (
                not order.get("picked_up_date")
                or date_in_window(order.get("picked_up_date"), inv.get("turnover_date"))
            )
        ]
        return candidates

    # Step 1: exact name + exact amount (+/-10 days).
    for order in orders:
        if order["id"] in used_orders:
            continue
        processed_steps += 1
        maybe_update_progress()
        candidates = find_candidates(order, name_exact)
        if not candidates:
            continue
        if order.get("picked_up_date") is None and len(candidates) != 1:
            continue
        inv = select_best_invoice(order, candidates)
        if not inv:
            continue
        conn.execute(
            "INSERT OR IGNORE INTO invoice_matches "
            "(order_id, invoice_id, score, status, method, matched_at) "
            "VALUES (?, ?, ?, ?, ?, datetime('now'))",
            (order["id"], inv["id"], 100, "auto", "exact"),
        )
        used_orders.add(order["id"])
        matched_invoice_ids.add(inv["id"])

    # Step 2: name within 1 char + exact amount (+/-10 days).
    for order in orders:
        if order["id"] in used_orders:
            continue
        processed_steps += 1
        maybe_update_progress()
        candidates = find_candidates(order, name_close)
        if not candidates:
            continue
        if order.get("picked_up_date") is None and len(candidates) != 1:
            continue
        inv = select_best_invoice(order, candidates)
        if not inv:
            continue
        conn.execute(
            "INSERT OR IGNORE INTO invoice_matches "
            "(order_id, invoice_id, score, status, method, matched_at) "
            "VALUES (?, ?, ?, ?, ?, datetime('now'))",
            (order["id"], inv["id"], 90, "auto", "close-name"),
        )
        used_orders.add(order["id"])
        matched_invoice_ids.add(inv["id"])

    # Store top candidates for unmatched orders.
    processed_steps += 1
    maybe_update_progress()

    remaining_invoices = [
        inv for inv in invoices
            if inv["id"] not in matched_invoice_ids
    ]
    for order in orders:
        if order["id"] in used_orders:
            continue
        processed_steps += 1
        maybe_update_progress()
        candidates = []
        for inv in candidate_invoices_for_order(order, remaining_invoices):
            score, reasons = score_match_with_reasons(order, inv)
            if score > 0:
                candidates.append((score, inv["id"], ",".join(reasons)))
        candidates.sort(reverse=True)
        top = candidates[:3]
        if not top:
            conn.execute(
                "INSERT OR IGNORE INTO order_flags (order_id, flag, note, created_at) "
                "VALUES (?, 'needs_invoice', NULL, datetime('now'))",
                (order["id"],),
            )
            continue
        conn.execute(
            "DELETE FROM order_flags WHERE order_id = ? AND flag = 'needs_invoice'",
            (order["id"],),
        )
        conn.execute("DELETE FROM invoice_candidates WHERE order_id = ?", (order["id"],))
        for score, inv_id, detail in top:
            conn.execute(
                "INSERT OR IGNORE INTO invoice_candidates "
                "(order_id, invoice_id, score, detail, method, created_at) "
                "VALUES (?, ?, ?, ?, ?, datetime('now'))",
                (order["id"], inv_id, score, detail, "fuzzy"),
            )

    if progress_task:
        update_task_progress(conn, progress_task, total_steps)
    conn.commit()


def list_review_matches(conn: sqlite3.Connection, return_rows: bool = False):
    rows = conn.execute(
        "SELECT m.id, m.score, o.sp_order_no, o.customer_name, o.picked_up_at, "
        "i.number, i.customer_name, i.turnover, i.amount_due "
        "FROM invoice_matches m "
        "JOIN orders o ON o.id = m.order_id "
        "JOIN invoices i ON i.id = m.invoice_id "
        "WHERE m.status = 'review' "
        "ORDER BY m.score DESC"
    ).fetchall()
    if not rows:
        print("Nema match-eva za pregled.")
        if return_rows:
            return [
                "match_id",
                "score",
                "sp_order_no",
                "order_name",
                "order_date",
                "invoice_no",
                "invoice_name",
                "invoice_date",
                "amount_due",
            ], []
        return
    for row in rows:
        print(
            f"match_id={row[0]} score={row[1]} "
            f"sp_order_no={row[2]} order_name={row[3]} order_date={row[4]} "
            f"invoice_no={row[5]} invoice_name={row[6]} invoice_date={row[7]} "
            f"amount_due={row[8]}"
        )
    if return_rows:
        return [
            "match_id",
            "score",
            "sp_order_no",
            "order_name",
            "order_date",
            "invoice_no",
            "invoice_name",
            "invoice_date",
            "amount_due",
        ], rows


def confirm_match(conn: sqlite3.Connection, match_id: int) -> None:
    row = conn.execute(
        "SELECT invoice_id FROM invoice_matches WHERE id = ?",
        (match_id,),
    ).fetchone()
    if not row:
        print("Match nije pronadjen.")
        return
    invoice_id = int(row[0])
    conn.execute(
        "UPDATE invoice_matches SET status = 'auto' WHERE id = ?",
        (match_id,),
    )
    conn.execute(
        "UPDATE invoices SET open_amount = 0, "
        "payment_amount = COALESCE(payment_amount, amount_due) "
        "WHERE id = ?",
        (invoice_id,),
    )
    conn.execute(
        "INSERT INTO action_log (action, ref_type, ref_id, note, created_at) "
        "VALUES ('confirm_match', 'invoice_match', ?, NULL, datetime('now'))",
        (match_id,),
    )
    conn.commit()
    print(f"Match potvrden: {match_id}.")


def apply_review_decisions(conn: sqlite3.Connection, path: Path) -> None:
    if path.suffix.lower() == ".xlsx":
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    if "match_id" not in df.columns:
        print("Fajl mora imati kolonu 'match_id'.")
        return
    if "confirm" not in df.columns and "needs_invoice" not in df.columns:
        print("Fajl mora imati kolonu 'confirm' ili 'needs_invoice' (1/0).")
        return
    confirmed = []
    needs_invoice = []
    if "confirm" in df.columns:
        confirmed = df[df["confirm"] == 1]["match_id"].dropna().astype(int).tolist()
    if "needs_invoice" in df.columns:
        needs_invoice = df[df["needs_invoice"] == 1]["match_id"].dropna().astype(int).tolist()
    if not confirmed:
        print("Nema potvrdenih match-eva.")
    for match_id in confirmed:
        confirm_match(conn, int(match_id))
    if needs_invoice:
        for match_id in needs_invoice:
            conn.execute(
                "UPDATE invoice_matches SET status = 'needs_invoice' WHERE id = ?",
                (int(match_id),),
            )
            conn.execute(
                "INSERT INTO action_log (action, ref_type, ref_id, note, created_at) "
                "VALUES ('needs_invoice', 'invoice_match', ?, NULL, datetime('now'))",
                (int(match_id),),
            )
        conn.commit()
    print(f"Potvrdeno ukupno: {len(confirmed)}")
    if needs_invoice:
        print(f"Oznaceno 'needs_invoice': {len(needs_invoice)}")

def report_unmatched_orders(conn: sqlite3.Connection, return_rows: bool = False):
    rows = conn.execute(
        "SELECT o.sp_order_no, o.customer_name, o.picked_up_at, o.status "
        "FROM orders o "
        "LEFT JOIN order_items oi ON oi.order_id = o.id "
        "WHERE o.id NOT IN (SELECT order_id FROM invoice_matches) "
        "AND o.id NOT IN (SELECT order_id FROM order_flags WHERE flag = 'needs_invoice') "
        "AND (o.status IS NULL OR lower(o.status) NOT LIKE '%otkazan%') "
        "AND (o.status IS NULL OR lower(o.status) NOT LIKE '%obradi%') "
        "AND date(substr(COALESCE(o.picked_up_at, o.created_at), 1, 10)) <= date('now', '-3 day') "
        "GROUP BY o.id "
        "HAVING SUM("
        "COALESCE(oi.cod_amount, 0) * (1 - COALESCE(oi.discount, 0) / 100.0) "
        "+ COALESCE(oi.addon_cod, 0) * (1 - COALESCE(oi.extra_discount, 0) / 100.0) "
        "- COALESCE(oi.advance_amount, 0) - COALESCE(oi.addon_advance, 0)"
        ") != 0 "
        "ORDER BY o.picked_up_at"
    ).fetchall()
    print(f"Neuparene narudzbe: {len(rows)}")
    for row in rows[:50]:
        print(f"{row[0]} | {row[1]} | {row[2]} | {row[3]}")
    if return_rows:
        return ["sp_order_no", "customer_name", "picked_up_at", "status"], rows


def report_conflicts(conn: sqlite3.Connection, return_rows: bool = False):
    order_rows = conn.execute(
        "SELECT o.id, o.sp_order_no, o.customer_name, o.picked_up_at, o.created_at, "
        "MAX(oi.cod_amount), MIN(oi.cod_amount), SUM(oi.cod_amount), "
        "MAX(oi.addon_cod), MIN(oi.addon_cod), SUM(oi.addon_cod), "
        "MAX(oi.advance_amount), MIN(oi.advance_amount), SUM(oi.advance_amount), "
        "MAX(oi.addon_advance), MIN(oi.addon_advance), SUM(oi.addon_advance) "
        "FROM orders o "
        "LEFT JOIN order_items oi ON oi.order_id = o.id "
        "GROUP BY o.id"
    ).fetchall()

    orders_by_date = {}
    orders = []
    for row in order_rows:
        cod = compute_order_component(row[5], row[6], row[7])
        addon = compute_order_component(row[8], row[9], row[10])
        advance = compute_order_component(row[11], row[12], row[13])
        addon_advance = compute_order_component(row[14], row[15], row[16])
        amount = compute_order_amount(cod, addon, advance, addon_advance)
        date_val = normalize_date(row[3] or row[4])
        order = {
            "id": int(row[0]),
            "sp_order_no": row[1],
            "name": row[2],
            "date": date_val,
            "amount": amount,
        }
        orders.append(order)
        if date_val:
            orders_by_date.setdefault(date_val, []).append(order)

    invoice_rows = conn.execute(
        "SELECT i.id, i.number, i.customer_name, i.turnover, i.amount_due, "
        "o.id, o.sp_order_no, o.customer_name "
        "FROM invoice_matches m "
        "JOIN invoices i ON i.id = m.invoice_id "
        "JOIN orders o ON o.id = m.order_id"
    ).fetchall()

    results = []
    for row in invoice_rows:
        inv_id = int(row[0])
        inv_no = row[1]
        inv_name = row[2]
        inv_date = normalize_date(row[3])
        inv_amount = row[4]
        matched_order_id = int(row[5])
        matched_sp = row[6]
        matched_name = row[7]

        if inv_date is None or inv_amount is None:
            continue
        candidates = []
        for offset in range(-10, 11):
            candidates.extend(orders_by_date.get(inv_date + timedelta(days=offset), []))
        for cand in candidates:
            if cand["id"] == matched_order_id:
                continue
            if cand["amount"] is None:
                continue
            try:
                if abs(float(cand["amount"]) - float(inv_amount)) > 0.01:
                    continue
            except (TypeError, ValueError):
                continue
            results.append(
                (
                    inv_no,
                    inv_name,
                    inv_date,
                    inv_amount,
                    matched_sp,
                    matched_name,
                    cand["sp_order_no"],
                    cand["name"],
                    cand["date"],
                    cand["amount"],
                )
            )

    print(f"Konflikti (racun vec uparen): {len(results)}")
    if return_rows:
        return [
            "invoice_no",
            "invoice_name",
            "invoice_date",
            "invoice_amount",
            "matched_sp_order",
            "matched_name",
            "candidate_sp_order",
            "candidate_name",
            "candidate_date",
            "candidate_amount",
        ], results


def report_nearest_invoice(conn: sqlite3.Connection, return_rows: bool = False):
    order_rows = conn.execute(
        "SELECT o.id, o.sp_order_no, o.customer_name, o.picked_up_at, o.created_at, "
        "MAX(oi.cod_amount), MIN(oi.cod_amount), SUM(oi.cod_amount), "
        "MAX(oi.addon_cod), MIN(oi.addon_cod), SUM(oi.addon_cod), "
        "MAX(oi.advance_amount), MIN(oi.advance_amount), SUM(oi.advance_amount), "
        "MAX(oi.addon_advance), MIN(oi.addon_advance), SUM(oi.addon_advance) "
        "FROM orders o "
        "LEFT JOIN order_items oi ON oi.order_id = o.id "
        "WHERE o.id NOT IN (SELECT order_id FROM invoice_matches) "
        "AND o.id NOT IN (SELECT order_id FROM order_flags WHERE flag = 'needs_invoice') "
        "AND (o.status IS NULL OR lower(o.status) NOT LIKE '%otkazan%') "
        "AND (o.status IS NULL OR lower(o.status) NOT LIKE '%obradi%') "
        "GROUP BY o.id"
    ).fetchall()

    order_ids = [int(row[0]) for row in order_rows]
    net_map = build_order_net_map(conn, order_ids)

    inv_rows = conn.execute(
        "SELECT i.id, i.number, i.customer_name, i.turnover, i.amount_due "
        "FROM invoices i"
    ).fetchall()
    invoices_by_date = {}
    invoices_no_date = []
    for row in inv_rows:
        inv = {
            "id": int(row[0]),
            "number": row[1],
            "name": row[2],
            "date": normalize_date(row[3]),
            "amount": row[4],
        }
        if inv["date"]:
            invoices_by_date.setdefault(inv["date"], []).append(inv)
        else:
            invoices_no_date.append(inv)

    results = []
    for row in order_rows:
        sp_no = row[1]
        name = row[2]
        display_date = row[3] or row[4]
        amount = net_map.get(int(row[0]))
        order_date = normalize_date(display_date)
        if amount is None or abs(amount) < 0.01:
            continue

        candidates = []
        if order_date:
            for offset in range(-10, 11):
                candidates.extend(invoices_by_date.get(order_date + timedelta(days=offset), []))
            candidates.extend(invoices_no_date)
        else:
            candidates = list(invoices_no_date)

        best = None
        best_diff = None
        for inv in candidates:
            if inv["amount"] is None:
                continue
            try:
                diff = abs(float(amount) - float(inv["amount"]))
            except (TypeError, ValueError):
                continue
            if best_diff is None or diff < best_diff:
                best_diff = diff
                best = inv

        if best is None:
            results.append((sp_no, name, display_date, amount, None, None, None, None))
        else:
            results.append(
                (
                    sp_no,
                    name,
                    display_date,
                    amount,
                    best["number"],
                    best["name"],
                    best["date"],
                    best_diff,
                )
            )

    print(f"Najblizi racuni: {len(results)}")
    if return_rows:
        return [
            "sp_order_no",
            "customer_name",
            "order_date",
            "order_amount",
            "invoice_no",
            "invoice_name",
            "invoice_date",
            "amount_diff",
        ], results


def report_unmatched_reasons(conn: sqlite3.Connection, return_rows: bool = False):
    rows = conn.execute(
        "SELECT o.id, o.sp_order_no, o.customer_name, o.picked_up_at, o.created_at, o.status, "
        "MAX(oi.cod_amount), MIN(oi.cod_amount), SUM(oi.cod_amount), "
        "MAX(oi.addon_cod), MIN(oi.addon_cod), SUM(oi.addon_cod), "
        "MAX(oi.advance_amount), MIN(oi.advance_amount), SUM(oi.advance_amount), "
        "MAX(oi.addon_advance), MIN(oi.addon_advance), SUM(oi.addon_advance) "
        "FROM orders o "
        "LEFT JOIN order_items oi ON oi.order_id = o.id "
        "WHERE o.id NOT IN (SELECT order_id FROM invoice_matches) "
        "AND o.id NOT IN (SELECT order_id FROM order_flags WHERE flag = 'needs_invoice') "
        "AND (o.status IS NULL OR lower(o.status) NOT LIKE '%otkazan%') "
        "GROUP BY o.id"
    ).fetchall()

    inv_rows = conn.execute(
        "SELECT i.id, i.customer_name, i.turnover, i.amount_due, "
        "CASE WHEN m.id IS NULL THEN 0 ELSE 1 END AS is_matched "
        "FROM invoices i "
        "LEFT JOIN invoice_matches m ON m.invoice_id = i.id"
    ).fetchall()
    matched_map = {
        int(row[0]): str(row[1])
        for row in conn.execute(
            "SELECT m.invoice_id, o.sp_order_no "
            "FROM invoice_matches m "
            "JOIN orders o ON o.id = m.order_id"
        ).fetchall()
    }
    invoices = []
    invoices_by_date = {}
    invoices_no_date = []
    for row in inv_rows:
        inv = {
            "id": int(row[0]),
            "name_norm": normalize_text(row[1]),
            "date": normalize_date(row[2]),
            "amount": row[3],
            "matched": bool(row[4]),
        }
        invoices.append(inv)
        if inv["date"]:
            invoices_by_date.setdefault(inv["date"], []).append(inv)
        else:
            invoices_no_date.append(inv)

    def amount_exact(a, b) -> bool:
        return amount_exact_strict(a, b)

    order_ids = [int(row[0]) for row in rows]
    net_map = build_order_net_map(conn, order_ids)

    results = []
    reason_counts = {}

    for row in rows:
        sp_no = row[1]
        name = row[2]
        picked_up_at = row[3]
        created_at = row[4]
        status = row[5]
        display_date = picked_up_at or created_at
        if is_cancelled_status(status) or is_in_progress_status(status):
            reason = "otkazano"
        else:
            amount = net_map.get(int(row[0]))
            order_date = normalize_date(display_date)

            if amount is None or abs(amount) < 0.01:
                continue
            else:
                if order_date and (date.today() - order_date).days <= 3:
                    continue
                if order_date:
                    date_candidates = []
                    for offset in range(-10, 11):
                        date_candidates.extend(
                            invoices_by_date.get(order_date + timedelta(days=offset), [])
                        )
                    date_candidates.extend(invoices_no_date)
                else:
                    date_candidates = invoices

                amount_candidates = [
                    inv for inv in date_candidates if amount_exact(amount, inv.get("amount"))
                ]
                unmatched_candidates = [inv for inv in amount_candidates if not inv.get("matched")]
                matched_candidates = [inv for inv in amount_candidates if inv.get("matched")]
                name_candidates = [
                    inv for inv in amount_candidates
                    if inv.get("name_norm") == normalize_text(name)
                ]
                name_unmatched = [inv for inv in name_candidates if not inv.get("matched")]
                name_matched = [inv for inv in name_candidates if inv.get("matched")]

                recent = False
                if order_date:
                    try:
                        recent = (date.today() - order_date).days <= 3
                    except Exception:
                        recent = False

                if not order_date:
                    reason = "nema_datuma_ima_iznos" if amount_candidates else "nema_datuma_nema_iznos"
                elif recent:
                    reason = "svjeza_narudzba"
                elif not amount_candidates:
                    reason = "nema_iznosa_u_prozoru"
                elif name_unmatched:
                    reason = "ima_kandidata_manual"
                elif name_matched:
                    other_orders = [
                        matched_map.get(inv["id"])
                        for inv in name_matched
                        if matched_map.get(inv["id"])
                    ]
                    if other_orders:
                        reason = f"racun_vec_uparen_ime_iznos: {', '.join(sorted(set(other_orders)))}"
                    else:
                        reason = "racun_vec_uparen_ime_iznos"
                elif unmatched_candidates:
                    reason = "ima_kandidata_manual"
                elif matched_candidates:
                    reason = "racun_vec_uparen"
                else:
                    reason = "nema_kandidata"

        reason_counts[reason] = reason_counts.get(reason, 0) + 1
        results.append((sp_no, name, display_date, reason))

    summary = sorted(reason_counts.items(), key=lambda x: x[1], reverse=True)
    print("Neuparene razlozi (top):")
    for reason, cnt in summary:
        print(f"{reason}: {cnt}")

    if return_rows:
        return ["sp_order_no", "customer_name", "picked_up_at", "reason"], results


def report_open_invoices(conn: sqlite3.Connection, return_rows: bool = False):
    rows = conn.execute(
        "SELECT number, customer_name, turnover, amount_due, open_amount "
        "FROM invoices "
        "WHERE open_amount IS NOT NULL AND open_amount > 0 "
        "ORDER BY turnover"
    ).fetchall()
    print(f"Otvorene fakture: {len(rows)}")
    for row in rows[:50]:
        print(f"{row[0]} | {row[1]} | {row[2]} | due={row[3]} | open={row[4]}")
    if return_rows:
        return ["number", "customer_name", "turnover", "amount_due", "open_amount"], rows


def report_returns(conn: sqlite3.Connection, return_rows: bool = False):
    rows = conn.execute(
        "SELECT status, COUNT(*) FROM returns GROUP BY status ORDER BY COUNT(*) DESC"
    ).fetchall()
    print("Povrati po statusu:")
    for row in rows:
        print(f"{row[0]}: {row[1]}")
    if return_rows:
        return ["status", "count"], rows


def report_needs_invoice(conn: sqlite3.Connection, return_rows: bool = False):
    rows = conn.execute(
        "SELECT m.id, o.sp_order_no, o.customer_name, o.picked_up_at, "
        "i.number, i.turnover, i.amount_due "
        "FROM invoice_matches m "
        "JOIN orders o ON o.id = m.order_id "
        "JOIN invoices i ON i.id = m.invoice_id "
        "WHERE m.status = 'needs_invoice' "
        "ORDER BY o.picked_up_at"
    ).fetchall()
    print(f"Potrebno kreirati racun: {len(rows)}")
    for row in rows[:50]:
        print(f"{row[0]} | {row[1]} | {row[2]} | {row[3]} | {row[4]} | {row[5]} | {row[6]}")
    if return_rows:
        return [
            "match_id",
            "sp_order_no",
            "customer_name",
            "picked_up_at",
            "invoice_no",
            "invoice_date",
            "amount_due",
        ], rows


def report_needs_invoice_orders(conn: sqlite3.Connection, return_rows: bool = False):
    rows = conn.execute(
        "SELECT o.sp_order_no, o.customer_name, o.picked_up_at, o.status "
        "FROM order_flags f "
        "JOIN orders o ON o.id = f.order_id "
        "WHERE f.flag = 'needs_invoice' "
        "ORDER BY o.picked_up_at"
    ).fetchall()
    print(f"Narudzbe bez kandidata (needs_invoice): {len(rows)}")
    for row in rows[:50]:
        print(f"{row[0]} | {row[1]} | {row[2]} | {row[3]}")
    if return_rows:
        return ["sp_order_no", "customer_name", "picked_up_at", "status"], rows


def report_no_value_orders(conn: sqlite3.Connection, return_rows: bool = False):
    rows = conn.execute(
        "SELECT o.sp_order_no, o.customer_name, o.picked_up_at, o.status "
        "FROM orders o "
        "LEFT JOIN order_items oi ON oi.order_id = o.id "
        "GROUP BY o.id "
        "HAVING SUM(COALESCE(oi.cod_amount, 0) + COALESCE(oi.addon_cod, 0) "
        "+ COALESCE(oi.advance_amount, 0) + COALESCE(oi.addon_advance, 0)) = 0 "
        "ORDER BY o.picked_up_at"
    ).fetchall()
    print(f"SP narudzbe bez vrijednosti: {len(rows)}")
    for row in rows[:50]:
        print(f"{row[0]} | {row[1]} | {row[2]} | {row[3]}")
    if return_rows:
        return ["sp_order_no", "customer_name", "picked_up_at", "status"], rows


def report_candidates(conn: sqlite3.Connection, return_rows: bool = False):
    rows = conn.execute(
        "SELECT o.sp_order_no, o.customer_name, o.picked_up_at, "
        "i.number, i.customer_name, i.turnover, i.amount_due, c.score, c.detail "
        "FROM invoice_candidates c "
        "JOIN orders o ON o.id = c.order_id "
        "JOIN invoices i ON i.id = c.invoice_id "
        "ORDER BY o.picked_up_at, c.score DESC"
    ).fetchall()
    print(f"Kandidati za neuparene: {len(rows)}")
    for row in rows[:50]:
        print(
            f"{row[0]} | {row[1]} | {row[2]} | {row[3]} | "
            f"{row[4]} | {row[5]} | {row[6]} | {row[7]} | {row[8]}"
        )
    if return_rows:
        return [
            "sp_order_no",
            "order_name",
            "order_date",
            "invoice_no",
            "invoice_name",
            "invoice_date",
            "amount_due",
            "score",
            "detail",
        ], rows


def report_storno(conn: sqlite3.Connection, return_rows: bool = False):
    rows = conn.execute(
        "SELECT s.storno_invoice_id, si.number, s.original_invoice_id, oi.number, "
        "s.storno_amount, s.original_amount, s.remaining_open, s.is_partial "
        "FROM invoice_storno s "
        "JOIN invoices si ON si.id = s.storno_invoice_id "
        "JOIN invoices oi ON oi.id = s.original_invoice_id "
        "ORDER BY s.is_partial DESC, si.number"
    ).fetchall()
    print(f"Storno veze: {len(rows)}")
    for row in rows[:50]:
        print(
            f"{row[1]} -> {row[3]} | storno={row[4]} | orig={row[5]} | open={row[6]} | partial={row[7]}"
        )
    if return_rows:
        return [
            "storno_invoice_no",
            "original_invoice_no",
            "storno_amount",
            "original_amount",
            "remaining_open",
            "is_partial",
        ], [(r[1], r[3], r[4], r[5], r[6], r[7]) for r in rows]


def report_bank_sp(conn: sqlite3.Connection, return_rows: bool = False):
    rows = conn.execute(
        "SELECT dtposted, amount, purpose, payee_name "
        "FROM bank_transactions "
        "WHERE benefit = 'credit' AND payee_name LIKE '%SLANJE PAKETA%' "
        "ORDER BY dtposted"
    ).fetchall()
    print(f"SP uplate na banku: {len(rows)}")
    for row in rows[:50]:
        print(f"{row[0]} | {row[1]} | {row[2]} | {row[3]}")
    if return_rows:
        return ["dtposted", "amount", "purpose", "payee_name"], rows


def report_bank_refunds(conn: sqlite3.Connection, return_rows: bool = False):
    rows = conn.execute(
        "SELECT dtposted, amount, purpose, payee_name "
        "FROM bank_transactions "
        "WHERE benefit = 'debit' AND (purpose LIKE '%Povrat%' OR purpose LIKE '%storno%') "
        "ORDER BY dtposted"
    ).fetchall()
    print(f"Refundacije/povrati (banka): {len(rows)}")
    for row in rows[:50]:
        print(f"{row[0]} | {row[1]} | {row[2]} | {row[3]}")
    if return_rows:
        return ["dtposted", "amount", "purpose", "payee_name"], rows


def extract_bank_refunds(conn: sqlite3.Connection) -> int:
    rows = conn.execute(
        "SELECT id, purpose, refnumber, payeerefnumber "
        "FROM bank_transactions "
        "WHERE benefit = 'debit' "
        "AND id NOT IN (SELECT bank_txn_id FROM bank_refunds)"
    ).fetchall()
    inserted = 0
    for row in rows:
        txn_id, purpose, refnumber, payeerefnumber = row
        reason = classify_refund_reason(purpose)
        if not reason:
            continue
        invoice_no, digits = extract_invoice_no_from_text(purpose)
        source = "purpose" if invoice_no else None
        if not invoice_no:
            invoice_no, digits = extract_invoice_no_from_text(refnumber)
            source = "refnumber" if invoice_no else None
        if not invoice_no:
            invoice_no, digits = extract_invoice_no_from_text(payeerefnumber)
            source = "payeerefnumber" if invoice_no else None
        cur = conn.execute(
            "INSERT OR IGNORE INTO bank_refunds ("
            "bank_txn_id, invoice_no, invoice_no_digits, invoice_no_source, reason, created_at"
            ") VALUES (?, ?, ?, ?, ?, datetime('now'))",
            (int(txn_id), invoice_no, digits, source, reason),
        )
        if cur.rowcount:
            inserted += 1
    conn.commit()
    print(f"Izvuceni povrati (banka): {inserted}")
    return inserted


def report_bank_refunds_extracted(conn: sqlite3.Connection, return_rows: bool = False):
    rows = conn.execute(
        "SELECT bt.dtposted, bt.amount, bt.purpose, bt.payee_name, "
        "br.invoice_no, br.invoice_no_digits, br.invoice_no_source, br.reason "
        "FROM bank_refunds br "
        "JOIN bank_transactions bt ON bt.id = br.bank_txn_id "
        "ORDER BY bt.dtposted"
    ).fetchall()
    print(f"Povrati iz izvoda (ekstrakt): {len(rows)}")
    for row in rows[:50]:
        print(
            f"{row[0]} | {row[1]} | {row[2]} | {row[3]} | "
            f"{row[4]} | {row[6]} | {row[7]}"
        )
    if return_rows:
        return [
            "dtposted",
            "amount",
            "purpose",
            "payee_name",
            "invoice_no",
            "invoice_no_digits",
            "invoice_no_source",
            "reason",
        ], rows

def report_bank_unmatched_sp(conn: sqlite3.Connection, return_rows: bool = False):
    rows = conn.execute(
        "SELECT dtposted, amount, purpose, payee_name "
        "FROM bank_transactions "
        "WHERE benefit = 'credit' AND payee_name LIKE '%SLANJE PAKETA%' "
        "AND id NOT IN (SELECT bank_txn_id FROM bank_matches) "
        "ORDER BY dtposted"
    ).fetchall()
    print(f"Neuparene SP uplate (banka): {len(rows)}")
    for row in rows[:50]:
        print(f"{row[0]} | {row[1]} | {row[2]} | {row[3]}")
    if return_rows:
        return ["dtposted", "amount", "purpose", "payee_name"], rows


def report_bank_unmatched_refunds(conn: sqlite3.Connection, return_rows: bool = False):
    rows = conn.execute(
        "SELECT dtposted, amount, purpose, payee_name "
        "FROM bank_transactions "
        "WHERE benefit = 'debit' AND (purpose LIKE '%Povrat%' OR purpose LIKE '%storno%') "
        "AND id NOT IN (SELECT bank_txn_id FROM bank_matches) "
        "ORDER BY dtposted"
    ).fetchall()
    print(f"Neuparene refundacije (banka): {len(rows)}")
    for row in rows[:50]:
        print(f"{row[0]} | {row[1]} | {row[2]} | {row[3]}")
    if return_rows:
        return ["dtposted", "amount", "purpose", "payee_name"], rows


def report_sp_vs_bank(conn: sqlite3.Connection, return_rows: bool = False):
    rows = conn.execute(
        "SELECT substr(o.picked_up_at, 1, 7) AS period, SUM(p.amount) AS sp_sum "
        "FROM payments p "
        "JOIN orders o ON o.sp_order_no = p.sp_order_no "
        "WHERE o.picked_up_at IS NOT NULL "
        "GROUP BY period"
    ).fetchall()
    bank_rows = conn.execute(
        "SELECT substr(dtposted, 1, 7) AS period, SUM(amount) AS bank_sum "
        "FROM bank_transactions "
        "WHERE benefit = 'credit' AND payee_name LIKE '%SLANJE PAKETA%' "
        "GROUP BY period"
    ).fetchall()
    bank_map = {r[0]: r[1] for r in bank_rows}
    merged = []
    for period, sp_sum in rows:
        bank_sum = bank_map.get(period)
        diff = None
        if sp_sum is not None and bank_sum is not None:
            diff = float(sp_sum) - float(bank_sum)
        merged.append((period, sp_sum, bank_sum, diff))
    print(f"SP vs banka periodi: {len(merged)}")
    for row in merged[:50]:
        print(f"{row[0]} | sp={row[1]} | bank={row[2]} | diff={row[3]}")
    if return_rows:
        return ["period", "sp_sum", "bank_sum", "diff"], merged


def close_invoices_from_confirmed_matches(conn: sqlite3.Connection) -> None:
    rows = conn.execute(
        "SELECT m.invoice_id FROM invoice_matches m WHERE m.status = 'auto'"
    ).fetchall()
    for (invoice_id,) in rows:
        conn.execute(
            "UPDATE invoices SET open_amount = 0, "
            "payment_amount = COALESCE(payment_amount, amount_due) "
            "WHERE id = ?",
            (int(invoice_id),),
        )
    conn.commit()


def reset_minimax_matches(conn: sqlite3.Connection) -> None:
    conn.execute("DELETE FROM invoice_matches")
    conn.execute("DELETE FROM invoice_candidates")
    conn.execute("DELETE FROM order_flags WHERE flag = 'needs_invoice'")
    conn.commit()


def report_alarms(conn: sqlite3.Connection, days: int = 7, return_rows: bool = False):
    rows = conn.execute(
        "SELECT o.sp_order_no, o.customer_name, o.picked_up_at, o.status "
        "FROM orders o "
        "LEFT JOIN order_items oi ON oi.order_id = o.id "
        "WHERE o.picked_up_at IS NOT NULL "
        "AND julianday('now') - julianday(o.picked_up_at) > ? "
        "AND o.id NOT IN (SELECT order_id FROM invoice_matches) "
        "AND (o.status IS NULL OR lower(o.status) NOT LIKE '%otkazan%') "
        "GROUP BY o.id "
        "HAVING SUM(COALESCE(oi.cod_amount, 0) + COALESCE(oi.addon_cod, 0) "
        "+ COALESCE(oi.advance_amount, 0) + COALESCE(oi.addon_advance, 0)) != 0 "
        "ORDER BY o.picked_up_at",
        (days,),
    ).fetchall()
    print(f"Alarmi (stare bez racuna): {len(rows)}")
    for row in rows[:50]:
        print(f"{row[0]} | {row[1]} | {row[2]} | {row[3]}")
    if return_rows:
        return ["sp_order_no", "customer_name", "picked_up_at", "status"], rows


def report_refunds_without_storno(conn: sqlite3.Connection, return_rows: bool = False):
    rows = conn.execute(
        "SELECT bt.dtposted, bt.amount, bt.purpose, bt.payee_name "
        "FROM bank_transactions bt "
        "LEFT JOIN bank_matches bm ON bm.bank_txn_id = bt.id AND bm.match_type = 'storno' "
        "WHERE bt.benefit = 'debit' AND (bt.purpose LIKE '%Povrat%' OR bt.purpose LIKE '%storno%') "
        "AND bm.id IS NULL "
        "ORDER BY bt.dtposted"
    ).fetchall()
    print(f"Refundacije bez storno racuna: {len(rows)}")
    for row in rows[:50]:
        print(f"{row[0]} | {row[1]} | {row[2]} | {row[3]}")
    if return_rows:
        return ["dtposted", "amount", "purpose", "payee_name"], rows


def report_order_amount_issues(conn: sqlite3.Connection, return_rows: bool = False):
    rows = conn.execute(
        "SELECT o.sp_order_no, o.customer_name, o.picked_up_at, "
        "SUM(COALESCE(oi.cod_amount, 0) + COALESCE(oi.addon_cod, 0)) AS base_sum, "
        "SUM(COALESCE(oi.advance_amount, 0) + COALESCE(oi.addon_advance, 0)) AS advance_sum, "
        "SUM(COALESCE(oi.cod_amount, 0) + COALESCE(oi.addon_cod, 0) - "
        "COALESCE(oi.advance_amount, 0) - COALESCE(oi.addon_advance, 0)) AS net_sum "
        "FROM orders o "
        "LEFT JOIN order_items oi ON oi.order_id = o.id "
        "GROUP BY o.id "
        "HAVING net_sum < 0 "
        "ORDER BY o.picked_up_at"
    ).fetchall()
    print(f"Izvjestaj suma (neto < 0): {len(rows)}")
    for row in rows[:50]:
        print(f"{row[0]} | {row[1]} | {row[2]} | base={row[3]} | adv={row[4]} | net={row[5]}")
    if return_rows:
        return ["sp_order_no", "customer_name", "picked_up_at", "base_sum", "advance_sum", "net_sum"], rows


def report_duplicate_customers(conn: sqlite3.Connection, return_rows: bool = False):
    rows = conn.execute(
        "SELECT customer_key, COUNT(*) AS cnt, MIN(picked_up_at), MAX(picked_up_at) "
        "FROM orders "
        "WHERE customer_key IS NOT NULL AND customer_key != '' "
        "GROUP BY customer_key "
        "HAVING COUNT(*) > 1 "
        "ORDER BY cnt DESC"
    ).fetchall()
    print(f"Dupli customer_key: {len(rows)}")
    for row in rows[:50]:
        print(f"{row[0]} | {row[1]} | {row[2]} -> {row[3]}")
    if return_rows:
        return ["customer_key", "count", "first_order", "last_order"], rows


def report_top_customers(conn: sqlite3.Connection, return_rows: bool = False):
    rows = conn.execute(
        "SELECT o.customer_key, COUNT(DISTINCT o.id) AS orders_cnt, "
        "SUM(COALESCE(oi.cod_amount, 0) + COALESCE(oi.addon_cod, 0) - "
        "COALESCE(oi.advance_amount, 0) - COALESCE(oi.addon_advance, 0)) AS net_total, "
        "MIN(o.picked_up_at), MAX(o.picked_up_at) "
        "FROM orders o "
        "LEFT JOIN order_items oi ON oi.order_id = o.id "
        "WHERE o.customer_key IS NOT NULL AND o.customer_key != '' "
        "GROUP BY o.customer_key "
        "ORDER BY net_total DESC"
    ).fetchall()
    print(f"Top kupci: {len(rows)}")
    for row in rows[:50]:
        print(f"{row[0]} | orders={row[1]} | net={row[2]} | {row[3]} -> {row[4]}")
    if return_rows:
        return ["customer_key", "orders_count", "net_total", "first_order", "last_order"], rows


def report_minimax_items(conn: sqlite3.Connection, return_rows: bool = False):
    rows = conn.execute(
        "SELECT sku, name, unit, mass_kg, closing_qty, "
        "closing_purchase_value, closing_sales_value, "
        "opening_qty, opening_purchase_value, incoming_qty, incoming_purchase_value, "
        "outgoing_qty, outgoing_sales_value "
        "FROM minimax_items "
        "ORDER BY sku"
    ).fetchall()
    print(f"Minimax artikli: {len(rows)}")
    enriched = []
    for row in rows[:50]:
        print(f"{row[0]} | {row[1]} | {row[2]} | {row[3]} | {row[4]}")
    for row in rows:
        sku, name, unit, mass_kg, closing_qty, closing_pv, closing_sv, \
            opening_qty, opening_pv, incoming_qty, incoming_pv, \
            outgoing_qty, outgoing_sv = row
        try:
            total_in_qty = (opening_qty or 0) + (incoming_qty or 0)
            avg_purchase = (opening_pv or 0) + (incoming_pv or 0)
            avg_purchase = avg_purchase / total_in_qty if total_in_qty else None
        except Exception:
            avg_purchase = None
        try:
            avg_sale = (outgoing_sv or 0) / (outgoing_qty or 0) if outgoing_qty else None
        except Exception:
            avg_sale = None
        margin = None
        if avg_purchase is not None and avg_sale is not None:
            try:
                margin = float(avg_sale) - float(avg_purchase)
            except Exception:
                margin = None
        enriched.append(
            (
                sku,
                name,
                unit,
                mass_kg,
                closing_qty,
                closing_pv,
                closing_sv,
                avg_purchase,
                avg_sale,
                margin,
            )
        )
    if return_rows:
        return [
            "sku",
            "name",
            "unit",
            "mass_kg",
            "closing_qty",
            "closing_purchase_value",
            "closing_sales_value",
            "avg_purchase_price",
            "avg_sale_price",
            "avg_margin",
        ], enriched


def report_category_sales(conn: sqlite3.Connection, return_rows: bool = False):
    rows = conn.execute(
        "SELECT oi.product_code, "
        "SUM(COALESCE(oi.cod_amount, 0) + COALESCE(oi.addon_cod, 0) - "
        "COALESCE(oi.advance_amount, 0) - COALESCE(oi.addon_advance, 0)) AS net_total, "
        "SUM(COALESCE(oi.qty, 0)) AS qty "
        "FROM order_items oi "
        "WHERE oi.product_code IS NOT NULL AND oi.product_code != '' "
        "GROUP BY oi.product_code"
    ).fetchall()
    by_cat = {}
    for sku, net_total, qty in rows:
        cat = kategorija_za_sifru(str(sku))
        cur = by_cat.get(cat, {"net_total": 0.0, "qty": 0.0})
        cur["net_total"] += float(net_total or 0)
        cur["qty"] += float(qty or 0)
        by_cat[cat] = cur
    merged = [
        (cat, vals["net_total"], vals["qty"]) for cat, vals in by_cat.items()
    ]
    merged.sort(key=lambda x: x[1], reverse=True)
    print(f"Prodaja po kategoriji: {len(merged)}")
    for row in merged[:50]:
        print(f"{row[0]} | net={row[1]} | qty={row[2]}")
    if return_rows:
        return ["category", "net_total", "qty"], merged


def report_category_returns(conn: sqlite3.Connection, return_rows: bool = False):
    rows = conn.execute(
        "SELECT o.id, o.status, oi.product_code, oi.qty "
        "FROM orders o "
        "LEFT JOIN order_items oi ON oi.order_id = o.id "
        "WHERE o.status IS NOT NULL"
    ).fetchall()
    by_cat = {}
    for _, status, sku, qty in rows:
        if not status:
            continue
        status_norm = normalize_text(status)
        if not status_norm.startswith("vrac"):
            continue
        if not sku:
            continue
        cat = kategorija_za_sifru(str(sku))
        by_cat[cat] = by_cat.get(cat, 0.0) + float(qty or 0)
    merged = [(cat, qty) for cat, qty in by_cat.items()]
    merged.sort(key=lambda x: x[1], reverse=True)
    print(f"Povrati po kategoriji: {len(merged)}")
    for row in merged[:50]:
        print(f"{row[0]} | qty={row[1]}")
    if return_rows:
        return ["category", "qty"], merged


def _unpicked_rows(conn: sqlite3.Connection, days: int | None = None, start: str | None = None, end: str | None = None):
    date_clause, params = _date_filter_clause("o.created_at", days, start, end)
    rows = conn.execute(
        "SELECT o.id, o.sp_order_no, o.customer_name, o.phone, o.email, o.city, "
        "o.status, o.created_at, o.customer_key, o.tracking_code, o.picked_up_at, o.delivered_at "
        "FROM orders o "
        "WHERE o.status IS NOT NULL " + date_clause,
        params,
    ).fetchall()
    return [row for row in rows if is_unpicked_status(row[6])]


def _order_items_for_orders(conn: sqlite3.Connection, order_ids: list[int]):
    if not order_ids:
        return []
    rows = []
    chunk_size = 900
    for i in range(0, len(order_ids), chunk_size):
        chunk = order_ids[i : i + chunk_size]
        placeholders = ",".join("?" * len(chunk))
        rows.extend(
            conn.execute(
                "SELECT order_id, product_code, qty, cod_amount, addon_cod, "
                "advance_amount, addon_advance "
                f"FROM order_items WHERE order_id IN ({placeholders})",
                chunk,
            ).fetchall()
        )
    return rows


def _net_simple(cod, addon, advance, addon_advance) -> float:
    return float(cod or 0) + float(addon or 0) - float(advance or 0) - float(addon_advance or 0)


def get_unpicked_stats(
    conn: sqlite3.Connection,
    days: int | None = None,
    start: str | None = None,
    end: str | None = None,
):
    rows = _unpicked_rows(conn, days, start, end)
    order_ids = [int(r[0]) for r in rows]
    items = _order_items_for_orders(conn, order_ids)
    order_net = {}
    for order_id, _, _, cod, addon, adv, addon_adv in items:
        order_net[order_id] = order_net.get(order_id, 0.0) + _net_simple(
            cod, addon, adv, addon_adv
        )
    total_lost = sum(order_net.values())

    groups = {}
    for _, _, name, phone, email, city, _, _, stored_key, *_ in rows:
        key = stored_key or compute_customer_key(phone, email, name, city)
        if not key:
            continue
        groups[key] = groups.get(key, 0) + 1
    repeat_customers = sum(1 for cnt in groups.values() if cnt >= 2)

    return {
        "unpicked_orders": len(order_ids),
        "lost_sales": float(total_lost or 0),
        "repeat_customers": repeat_customers,
    }


def match_bank_sp_payments(
    conn: sqlite3.Connection,
    day_tolerance: int = 2,
    progress_task: str | None = None,
    start_at: int = 0,
    total: int | None = None,
):
    txns = conn.execute(
        "SELECT id, dtposted, amount FROM bank_transactions "
        "WHERE benefit = 'credit' AND payee_name LIKE '%SLANJE PAKETA%' "
        "AND id NOT IN (SELECT bank_txn_id FROM bank_matches)"
    ).fetchall()
    payments = conn.execute(
        "SELECT p.id, p.amount, o.picked_up_at "
        "FROM payments p "
        "LEFT JOIN orders o ON o.sp_order_no = p.sp_order_no"
    ).fetchall()
    processed = start_at
    last_progress = start_at
    progress_every = 10

    def maybe_update_progress():
        nonlocal last_progress
        if not progress_task:
            return
        if processed - last_progress >= progress_every or (total and processed >= total):
            update_task_progress(conn, progress_task, processed)
            last_progress = processed

    for txn_id, dtposted, amount in txns:
        txn_date = normalize_date(dtposted)
        best = None
        best_score = 0
        best_method = "amount"
        for pay_id, pay_amount, picked_up_at in payments:
            if pay_amount is None or amount is None:
                continue
            try:
                amount_diff = abs(float(pay_amount) - float(amount))
            except (TypeError, ValueError):
                continue
            if amount_diff <= 0.01:
                score = 60
            elif amount_diff <= 2.0:
                score = 40
            else:
                continue
            pay_date = normalize_date(picked_up_at)
            if txn_date and pay_date:
                day_diff = abs((txn_date - pay_date).days)
                if day_diff <= day_tolerance:
                    score += 20
                    best_method = "amount+date"
            if score > best_score:
                best_score = score
                best = pay_id
        if best is not None:
            conn.execute(
                "INSERT OR IGNORE INTO bank_matches "
                "(bank_txn_id, match_type, ref_id, score, method, matched_at) "
                "VALUES (?, 'sp_payment', ?, ?, ?, datetime('now'))",
                (txn_id, best, best_score, best_method),
            )
        processed += 1
        maybe_update_progress()
    conn.commit()
    if progress_task:
        update_task_progress(conn, progress_task, processed)
        return processed
    return None


def match_bank_refunds(
    conn: sqlite3.Connection,
    progress_task: str | None = None,
    start_at: int = 0,
    total: int | None = None,
):
    txns = conn.execute(
        "SELECT id, purpose, refnumber, payeerefnumber FROM bank_transactions "
        "WHERE benefit = 'debit' AND (purpose LIKE '%Povrat%' OR purpose LIKE '%storno%') "
        "AND id NOT IN (SELECT bank_txn_id FROM bank_matches)"
    ).fetchall()
    processed = start_at
    last_progress = start_at
    progress_every = 10

    def maybe_update_progress():
        nonlocal last_progress
        if not progress_task:
            return
        if processed - last_progress >= progress_every or (total and processed >= total):
            update_task_progress(conn, progress_task, processed)
            last_progress = processed

    for txn_id, purpose, refnumber, payeerefnumber in txns:
        text = " ".join([str(purpose or ""), str(refnumber or ""), str(payeerefnumber or "")])
        invoice_no = extract_invoice_number_from_text(text)
        if not invoice_no:
            processed += 1
            maybe_update_progress()
            continue
        row = conn.execute(
            "SELECT id FROM invoices WHERE number = ?",
            (invoice_no,),
        ).fetchone()
        if not row:
            processed += 1
            maybe_update_progress()
            continue
        conn.execute(
            "INSERT OR IGNORE INTO bank_matches "
            "(bank_txn_id, match_type, ref_id, score, method, matched_at) "
            "VALUES (?, 'storno', ?, 100, 'purpose', datetime('now'))",
            (txn_id, int(row[0])),
        )
        processed += 1
        maybe_update_progress()
    conn.commit()
    if progress_task:
        update_task_progress(conn, progress_task, processed)
        return processed
    return None


def run_match_minimax_process(db_path: str) -> None:
    conn = connect_db(Path(db_path))
    init_db(conn)
    match_minimax(conn, progress_task="match_minimax")
    conn.close()


def run_match_bank_process(db_path: str, day_tolerance: int = 2) -> None:
    conn = connect_db(Path(db_path))
    init_db(conn)
    task = "match_bank"
    credit_count = conn.execute(
        "SELECT COUNT(*) FROM bank_transactions "
        "WHERE benefit = 'credit' AND payee_name LIKE '%SLANJE PAKETA%' "
        "AND id NOT IN (SELECT bank_txn_id FROM bank_matches)"
    ).fetchone()[0]
    debit_count = conn.execute(
        "SELECT COUNT(*) FROM bank_transactions "
        "WHERE benefit = 'debit' AND (purpose LIKE '%Povrat%' OR purpose LIKE '%storno%') "
        "AND id NOT IN (SELECT bank_txn_id FROM bank_matches)"
    ).fetchone()[0]
    total = int(credit_count or 0) + int(debit_count or 0) + 2
    set_task_progress(conn, task, total)
    update_task_progress(conn, task, 1)
    processed = 1
    processed = match_bank_sp_payments(
        conn,
        day_tolerance,
        progress_task=task,
        start_at=processed,
        total=total,
    ) or processed
    processed = match_bank_refunds(
        conn,
        progress_task=task,
        start_at=processed,
        total=total,
    ) or processed
    update_task_progress(conn, task, total)
    conn.close()


def run_tracking_process(db_path: str, batch_size: int = 20, force_refresh: int = 0) -> None:
    conn = connect_db(Path(db_path))
    init_db(conn)
    ensure_customer_keys(conn)
    update_unpicked_tracking(
        conn,
        batch_size=batch_size,
        progress_task="tracking",
        force_refresh=bool(force_refresh),
    )
    conn.close()


def report_unmatched_with_candidates(conn: sqlite3.Connection, return_rows: bool = False):
    rows = conn.execute(
        "SELECT o.sp_order_no, o.customer_name, o.picked_up_at, o.status, "
        "i.number, i.customer_name, i.turnover, i.amount_due, c.score, c.detail "
        "FROM orders o "
        "LEFT JOIN order_items oi ON oi.order_id = o.id "
        "LEFT JOIN invoice_candidates c ON c.order_id = o.id "
        "LEFT JOIN invoices i ON i.id = c.invoice_id "
        "WHERE o.id NOT IN (SELECT order_id FROM invoice_matches) "
        "AND o.id NOT IN (SELECT order_id FROM order_flags WHERE flag = 'needs_invoice') "
        "AND (o.status IS NULL OR lower(o.status) NOT LIKE '%otkazan%') "
        "AND (o.status IS NULL OR lower(o.status) NOT LIKE '%obradi%') "
        "AND date(substr(COALESCE(o.picked_up_at, o.created_at), 1, 10)) <= date('now', '-3 day') "
        "GROUP BY o.id, c.id "
        "HAVING SUM("
        "COALESCE(oi.cod_amount, 0) * (1 - COALESCE(oi.discount, 0) / 100.0) "
        "+ COALESCE(oi.addon_cod, 0) * (1 - COALESCE(oi.extra_discount, 0) / 100.0) "
        "- COALESCE(oi.advance_amount, 0) - COALESCE(oi.addon_advance, 0)"
        ") != 0 "
        "ORDER BY o.picked_up_at, c.score DESC"
    ).fetchall()
    print(f"Neuparene + kandidati: {len(rows)}")
    for row in rows[:50]:
        print(
            f"{row[0]} | {row[1]} | {row[2]} | {row[3]} | "
            f"{row[4]} | {row[5]} | {row[6]} | {row[7]} | {row[8]} | {row[9]}"
        )
    if return_rows:
        return [
            "sp_order_no",
            "order_name",
            "order_date",
            "status",
            "candidate_invoice_no",
            "candidate_invoice_name",
            "candidate_invoice_date",
            "candidate_amount_due",
            "candidate_score",
            "candidate_detail",
        ], rows


def report_unmatched_with_candidates_grouped(conn: sqlite3.Connection, return_rows: bool = False):
    cols, rows = report_unmatched_with_candidates(conn, return_rows=True)
    grouped_rows = []
    last_key = None
    for row in rows:
        key = (row[0], row[1], row[2], row[3])
        if key != last_key:
            if last_key is not None:
                grouped_rows.append([""] * len(cols))
            grouped_rows.append(
                [
                    row[0],
                    row[1],
                    row[2],
                    row[3],
                    "",
                    "",
                    "",
                    "",
                    "",
                ]
            )
        grouped_rows.append(list(row))
        last_key = key
    if return_rows:
        return cols, grouped_rows
    print(f"Neuparene + kandidati (grupisano): {len(grouped_rows)}")


def write_report(columns, rows, out_dir: Path, name: str, fmt: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"{name}_{stamp}.{fmt}"
    df = pd.DataFrame(rows, columns=columns)
    if fmt == "xlsx":
        df.to_excel(out_path, index=False)
    else:
        df.to_csv(out_path, index=False)
    return out_path


def _date_expr(column: str) -> str:
    return (
        "CASE "
        f"WHEN {column} IS NULL THEN NULL "
        f"WHEN substr({column}, 3, 1) = '.' AND substr({column}, 6, 1) = '.' "
        f"THEN date(substr({column}, 7, 4) || '-' || substr({column}, 4, 2) || '-' || substr({column}, 1, 2)) "
        f"ELSE date(substr({column}, 1, 10)) "
        "END"
    )


def _date_filter_clause(column: str, days: int | None, start: str | None = None, end: str | None = None):
    params = []
    clauses = []
    expr = _date_expr(column)
    if start:
        clauses.append(f"{expr} >= date(?)")
        params.append(start)
    if end:
        clauses.append(f"{expr} <= date(?)")
        params.append(end)
    if not start and not end and days:
        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        clauses.append(f"{expr} >= date(?)")
        params.append(start_date)
    if not clauses:
        return "", []
    clause = " AND " + " AND ".join(clauses)
    return clause, params


def get_top_customers(conn: sqlite3.Connection, limit: int = 5, days: int | None = None, start: str | None = None, end: str | None = None):
    date_clause, params = _date_filter_clause("o.picked_up_at", days, start, end)
    rows = conn.execute(
        "SELECT COALESCE(NULLIF(TRIM(MAX(o.customer_name)), ''), o.customer_key) AS display_name, "
        "COUNT(DISTINCT o.id) AS orders_cnt, "
        "SUM(COALESCE(oi.cod_amount, 0) + COALESCE(oi.addon_cod, 0) - "
        "COALESCE(oi.advance_amount, 0) - COALESCE(oi.addon_advance, 0)) AS net_total "
        "FROM orders o "
        "LEFT JOIN order_items oi ON oi.order_id = o.id "
        "WHERE o.customer_key IS NOT NULL AND o.customer_key != '' "
        "AND o.picked_up_at IS NOT NULL "
        "AND (o.status IS NULL OR (o.status NOT LIKE '%Vra\u0107eno%' AND o.status NOT LIKE '%Vraceno%')) "
        + date_clause +
        " GROUP BY o.customer_key "
        "ORDER BY net_total DESC "
        "LIMIT ?",
        (*params, limit),
    ).fetchall()
    return rows


def get_top_products(conn: sqlite3.Connection, limit: int = 10, days: int | None = None, start: str | None = None, end: str | None = None):
    date_clause, params = _date_filter_clause("o.picked_up_at", days, start, end)
    rows = conn.execute(
        "SELECT oi.product_code, "
        "SUM(COALESCE(oi.qty, 0)) AS total_qty, "
        "SUM(COALESCE(oi.cod_amount, 0) + COALESCE(oi.addon_cod, 0) - "
        "COALESCE(oi.advance_amount, 0) - COALESCE(oi.addon_advance, 0)) AS net_total "
        "FROM order_items oi "
        "JOIN orders o ON o.id = oi.order_id "
        "WHERE oi.product_code IS NOT NULL AND oi.product_code != '' "
        "AND o.picked_up_at IS NOT NULL "
        "AND (o.status IS NULL OR (o.status NOT LIKE '%Vra\u0107eno%' AND o.status NOT LIKE '%Vraceno%')) "
        + date_clause +
        " GROUP BY oi.product_code "
        "ORDER BY net_total DESC "
        "LIMIT ?",
        (*params, limit),
    ).fetchall()
    return rows


def get_kpis(conn: sqlite3.Connection, days: int | None = None, start: str | None = None, end: str | None = None):
    date_clause, params = _date_filter_clause("o.picked_up_at", days, start, end)
    total_orders = conn.execute(
        "SELECT COUNT(*) FROM orders o "
        "WHERE o.picked_up_at IS NOT NULL "
        "AND (o.status IS NULL OR (o.status NOT LIKE '%Vra\u0107eno%' AND o.status NOT LIKE '%Vraceno%')) "
        + date_clause,
        params,
    ).fetchone()[0]
    total_revenue = conn.execute(
        "SELECT SUM("
        "COALESCE(oi.cod_amount, 0) * (1 - COALESCE(oi.discount, 0) / 100.0) "
        "+ COALESCE(oi.addon_cod, 0) * (1 - COALESCE(oi.extra_discount, 0) / 100.0) "
        "- COALESCE(oi.advance_amount, 0) - COALESCE(oi.addon_advance, 0)"
        ") "
        "FROM orders o "
        "JOIN order_items oi ON oi.order_id = o.id "
        "WHERE o.picked_up_at IS NOT NULL "
        "AND (o.status IS NULL OR (o.status NOT LIKE '%Vra\u0107eno%' AND o.status NOT LIKE '%Vraceno%')) "
        + date_clause,
        params,
    ).fetchone()[0]
    unpicked_clause, unpicked_params = _date_filter_clause("o.created_at", days, start, end)
    total_unpicked = conn.execute(
        "SELECT COUNT(*) FROM orders o "
        "WHERE o.status LIKE '%Vra\u0107eno%' " + unpicked_clause,
        unpicked_params,
    ).fetchone()[0]
    unmatched = conn.execute(
        "SELECT COUNT(*) FROM orders o "
        "LEFT JOIN order_items oi ON oi.order_id = o.id "
        "WHERE o.id NOT IN (SELECT order_id FROM invoice_matches) "
        "AND o.id NOT IN (SELECT order_id FROM order_flags WHERE flag = 'needs_invoice') "
        "AND (o.status IS NULL OR lower(o.status) NOT LIKE '%otkazan%') "
        "AND (o.status IS NULL OR lower(o.status) NOT LIKE '%obradi%') "
        "AND date(substr(COALESCE(o.picked_up_at, o.created_at), 1, 10)) <= date('now', '-3 day') "
        "AND o.picked_up_at IS NOT NULL " + date_clause +
        " GROUP BY o.id "
        "HAVING SUM("
        "COALESCE(oi.cod_amount, 0) * (1 - COALESCE(oi.discount, 0) / 100.0) "
        "+ COALESCE(oi.addon_cod, 0) * (1 - COALESCE(oi.extra_discount, 0) / 100.0) "
        "- COALESCE(oi.advance_amount, 0) - COALESCE(oi.addon_advance, 0)"
        ") != 0",
        params,
    ).fetchall()
    unmatched_count = len(unmatched)
    return {
        "total_orders": total_orders or 0,
        "total_revenue": float(total_revenue or 0),
        "total_returns": total_unpicked or 0,
        "unmatched": unmatched_count,
    }


def get_sp_bank_monthly(conn: sqlite3.Connection, days: int | None = None, start: str | None = None, end: str | None = None):
    date_clause, params = _date_filter_clause("dtposted", days, start, end)
    rows = conn.execute(
        "SELECT substr(dtposted, 1, 7) AS period, "
        "SUM(CASE WHEN benefit = 'credit' "
        "AND lower(COALESCE(purpose, '')) NOT LIKE '%pozajmica%' "
        "THEN amount ELSE 0 END) AS income, "
        "("
        "SUM(CASE WHEN benefit = 'debit' "
        "AND lower(COALESCE(purpose, '')) NOT LIKE '%kupoprodaja deviza%' "
        "AND COALESCE(purposecode, '') != '286' "
        "AND lower(COALESCE(purpose, '')) NOT LIKE '%carin%' "
        "THEN amount ELSE 0 END) "
        "+ SUM(CASE WHEN benefit = 'debit' "
        "AND lower(COALESCE(purpose, '')) LIKE '%carin%' "
        "THEN amount * 0.20 ELSE 0 END)"
        ") AS expense "
        "FROM bank_transactions "
        "WHERE dtposted IS NOT NULL " + date_clause +
        " GROUP BY period "
        "ORDER BY period",
        params,
    ).fetchall()
    return rows


def _normalize_expense_key(text: str) -> str:
    value = re.sub(r"\s+", " ", (text or "").strip().lower())
    return value or "nepoznato"


def _expense_category(payee_name: str | None, purpose: str | None) -> str:
    text = normalize_text_loose(" ".join([str(payee_name or ""), str(purpose or "")]))
    if "slanje paketa" in text:
        return "Slanje Paketa"
    if (
        "svrha doprinosi" in text
        or "uplata poreza i doprinosa po odbitku" in text
        or "doprinosi" in text
    ):
        return "Doprinosi za socijalno osiguranje"
    if "840-0000714112843-10" in text:
        return "PDV"
    if "placanje pdv" in text or "plaćanje pdv" in text:
        return "PDV"
    label = (payee_name or "").strip()
    if not label:
        label = (purpose or "").strip()
    return label or "Nepoznato"


def _short_expense_name(label: str) -> str:
    if not label:
        return "Nepoznato"
    base = label.split(",")[0].strip()
    return base or label


def _extract_month_year(text: str | None) -> str | None:
    if not text:
        return None
    match = re.search(r"\b(0[1-9]|1[0-2])\.(20\d{2})\b", text)
    if not match:
        return None
    month, year = match.group(1), match.group(2)
    return f"{year}-{month}"


def _effective_expense_period(dtposted: str | None, purpose: str | None, category: str | None) -> str | None:
    explicit = _extract_month_year(purpose)
    if explicit:
        return explicit
    if not dtposted:
        return None
    try:
        dt = pd.to_datetime(dtposted, errors="coerce")
    except Exception:
        return None
    if pd.isna(dt):
        return None
    if category == "PDV":
        dt = dt - pd.DateOffset(months=1)
    return dt.strftime("%Y-%m")


def _is_forex_expense(purpose: str | None, purposecode: str | None) -> bool:
    code = str(purposecode or "").strip()
    if code == "286":
        return True
    text = normalize_text_loose(purpose)
    return "kupoprodaja deviza" in text


def _expense_amount(purpose: str | None, purposecode: str | None, amount: float) -> float | None:
    if _is_forex_expense(purpose, purposecode):
        return None
    text = normalize_text_loose(purpose)
    if "carin" in text:
        return amount * 0.20
    return amount


def get_expense_summary(
    conn: sqlite3.Connection,
    days: int | None = None,
    start: str | None = None,
    end: str | None = None,
    year: str | None = None,
    month: str | None = None,
) -> dict:
    date_clause, params = _date_filter_clause("dtposted", days, start, end)
    rows = conn.execute(
        "SELECT dtposted, amount, purpose, purposecode, payee_name "
        "FROM bank_transactions "
        "WHERE benefit = 'debit' AND dtposted IS NOT NULL "
        + date_clause,
        params,
    ).fetchall()
    total = 0.0
    totals: dict[str, float] = {}
    display_names: dict[str, str] = {}
    monthly: dict[str, dict[str, float]] = {}
    for dtposted, amount, purpose, purposecode, payee_name in rows:
        try:
            amt = float(amount or 0)
        except (TypeError, ValueError):
            continue
        final_amount = _expense_amount(purpose, purposecode, amt)
        if final_amount is None:
            continue
        label = _expense_category(payee_name, purpose)
        key = _normalize_expense_key(label)
        period = _effective_expense_period(dtposted, purpose, label)
        if year and (not period or not period.startswith(year)):
            continue
        if month and (not period or len(period) < 7 or period[5:7] != month):
            continue
        totals[key] = totals.get(key, 0.0) + final_amount
        if key not in display_names:
            display_names[key] = label
        total += final_amount
        if period:
            if period not in monthly:
                monthly[period] = {}
            monthly[period][key] = monthly[period].get(key, 0.0) + final_amount
    return {
        "total": total,
        "totals": totals,
        "display_names": display_names,
        "monthly": monthly,
    }


def _refund_rows(conn: sqlite3.Connection, days: int | None = None, start: str | None = None, end: str | None = None):
    date_clause, params = _date_filter_clause("bt.dtposted", days, start, end)
    rows = conn.execute(
        "SELECT br.bank_txn_id, bt.dtposted, bt.amount, bt.payee_name, bt.purpose, "
        "br.invoice_no, br.invoice_no_digits "
        "FROM bank_refunds br "
        "JOIN bank_transactions bt ON bt.id = br.bank_txn_id "
        "WHERE bt.dtposted IS NOT NULL " + date_clause +
        " ORDER BY bt.dtposted",
        params,
    ).fetchall()
    return rows


def get_refund_top_customers(
    conn: sqlite3.Connection,
    limit: int = 5,
    days: int | None = None,
    start: str | None = None,
    end: str | None = None,
):
    rows = _refund_rows(conn, days, start, end)
    if not rows:
        return []
    inv_by_number = {}
    inv_by_digits = {}
    for inv_id, inv_no, name, note, basis in conn.execute(
        "SELECT id, number, customer_name, note, basis FROM invoices"
    ).fetchall():
        inv_by_number[str(inv_no or "")] = (int(inv_id), name)
        digits = invoice_digits(inv_no)
        if digits:
            inv_by_digits.setdefault(digits, set()).add((int(inv_id), name))
        for text in (note, basis):
            inv_text, digits_text = extract_invoice_no_from_text(text or "")
            if digits_text:
                inv_by_digits.setdefault(digits_text, set()).add((int(inv_id), name))

    order_by_invoice = {
        int(inv_id): int(order_id)
        for inv_id, order_id in conn.execute(
            "SELECT invoice_id, order_id FROM invoice_matches"
        ).fetchall()
    }
    order_name = {
        int(oid): name
        for oid, name in conn.execute("SELECT id, customer_name FROM orders").fetchall()
    }

    totals = {}
    amounts = {}
    for _, _, amount, payee_name, _, inv_no, inv_digits in rows:
        key = None
        if inv_no and str(inv_no) in inv_by_number:
            inv_id, inv_name = inv_by_number[str(inv_no)]
            order_id = order_by_invoice.get(inv_id)
            key = order_name.get(order_id) or inv_name
        elif inv_digits and inv_digits in inv_by_digits and len(inv_by_digits[inv_digits]) == 1:
            inv_id, inv_name = next(iter(inv_by_digits[inv_digits]))
            order_id = order_by_invoice.get(inv_id)
            key = order_name.get(order_id) or inv_name
        if not key:
            key = payee_name or "Nepoznato"
        totals[key] = totals.get(key, 0) + 1
        amounts[key] = amounts.get(key, 0.0) + float(amount or 0)
    ranked = sorted(totals.items(), key=lambda x: x[1], reverse=True)
    return [(name, cnt, amounts.get(name, 0.0)) for name, cnt in ranked[:limit]]


def get_refund_top_items(
    conn: sqlite3.Connection,
    limit: int = 5,
    days: int | None = None,
    start: str | None = None,
    end: str | None = None,
):
    totals = build_refund_item_totals(conn, days, start, end)
    ranked = sorted(totals.items(), key=lambda x: x[1], reverse=True)
    return ranked[:limit]


def get_refund_top_categories(
    conn: sqlite3.Connection,
    limit: int = 5,
    days: int | None = None,
    start: str | None = None,
    end: str | None = None,
):
    totals = build_refund_item_totals(conn, days, start, end)
    if not totals:
        return []
    by_cat = {}
    for sku, qty in totals.items():
        cat = kategorija_za_sifru(str(sku))
        by_cat[cat] = by_cat.get(cat, 0.0) + float(qty or 0)
    ranked = sorted(by_cat.items(), key=lambda x: x[1], reverse=True)
    return ranked[:limit]


def _pick_display_name(name_counts: dict) -> str:
    if not name_counts:
        return ""
    return max(name_counts.items(), key=lambda x: (x[1], len(x[0])))[0]


def get_unpicked_customer_groups(
    conn: sqlite3.Connection,
    limit: int = 5,
    days: int | None = None,
    start: str | None = None,
    end: str | None = None,
):
    rows = _unpicked_rows(conn, days, start, end)
    if not rows:
        return [], []
    groups = {}
    for row in rows:
        sp_order_no = row[1]
        name = row[2]
        phone = row[3]
        email = row[4]
        city = row[5]
        stored_key = row[8] if len(row) > 8 else None
        key = stored_key or compute_customer_key(phone, email, name, city)
        if not key:
            continue
        group = groups.setdefault(
            key,
            {"count": 0, "names": {}, "phones": set(), "emails": set(), "orders": set()},
        )
        group["count"] += 1
        if name:
            group["names"][name] = group["names"].get(name, 0) + 1
        phone_norm = normalize_phone(phone)
        if phone_norm:
            group["phones"].add(phone_norm)
        if email:
            email_text = str(email).strip()
            if "@" in email_text:
                group["emails"].add(email_text)
            else:
                alt_phone = normalize_phone(email_text)
                if alt_phone:
                    group["phones"].add(alt_phone)
        if sp_order_no:
            group["orders"].add(str(sp_order_no))

    ranked = sorted(groups.items(), key=lambda x: x[1]["count"], reverse=True)
    top = []
    details = []
    for key, info in ranked:
        key_display = key
        if key.startswith("phone:"):
            key_display = key.replace("phone:", "", 1)
        elif key.startswith("email:"):
            key_display = key.replace("email:", "", 1)
        name = _pick_display_name(info["names"]) or key_display
        top.append((name, info["count"]))
        names_list = ", ".join(sorted(info["names"].keys()))
        phones_list = ", ".join(sorted(info["phones"]))
        emails_list = ", ".join(sorted(info["emails"]))
        details.append(
            (key_display, info["count"], names_list, phones_list, emails_list, len(info["orders"]))
        )
        if len(details) >= 50:
            break
    return top[:limit], details[:50]


def get_unpicked_top_items(
    conn: sqlite3.Connection,
    limit: int | None = 5,
    days: int | None = None,
    start: str | None = None,
    end: str | None = None,
):
    rows = _unpicked_rows(conn, days, start, end)
    order_ids = [int(r[0]) for r in rows]
    items = _order_items_for_orders(conn, order_ids)
    totals = {}
    for _, sku, qty, cod, addon, adv, addon_adv in items:
        if not sku:
            continue
        entry = totals.get(sku, {"qty": 0.0, "net": 0.0})
        entry["qty"] += float(qty or 0)
        entry["net"] += _net_simple(cod, addon, adv, addon_adv)
        totals[sku] = entry
    ranked = sorted(totals.items(), key=lambda x: x[1]["qty"], reverse=True)
    if limit is None:
        return [(sku, vals["qty"], vals["net"]) for sku, vals in ranked]
    return [(sku, vals["qty"], vals["net"]) for sku, vals in ranked[:limit]]


def get_unpicked_category_totals(
    conn: sqlite3.Connection,
    days: int | None = None,
    start: str | None = None,
    end: str | None = None,
):
    rows = _unpicked_rows(conn, days, start, end)
    order_ids = [int(r[0]) for r in rows]
    items = _order_items_for_orders(conn, order_ids)
    totals = {}
    for _, sku, qty, cod, addon, adv, addon_adv in items:
        if not sku:
            continue
        cat = kategorija_za_sifru(str(sku))
        entry = totals.get(cat, {"qty": 0.0, "net": 0.0})
        entry["qty"] += float(qty or 0)
        entry["net"] += _net_simple(cod, addon, adv, addon_adv)
        totals[cat] = entry
    ranked = sorted(totals.items(), key=lambda x: x[1]["qty"], reverse=True)
    return [(cat, vals["qty"], vals["net"]) for cat, vals in ranked]


def get_unpicked_orders_list(
    conn: sqlite3.Connection,
    days: int | None = None,
    start: str | None = None,
    end: str | None = None,
):
    rows = _unpicked_rows(conn, days, start, end)
    order_ids = [int(r[0]) for r in rows]
    items = _order_items_for_orders(conn, order_ids)
    order_net = {}
    for order_id, _, _, cod, addon, adv, addon_adv in items:
        order_net[order_id] = order_net.get(order_id, 0.0) + _net_simple(
            cod, addon, adv, addon_adv
        )
    result = []
    for row in rows:
        (
            order_id,
            sp_order_no,
            name,
            phone,
            email,
            city,
            status,
            created_at,
            _,
            tracking_code,
            picked_up_at,
            delivered_at,
        ) = row
        result.append(
            (
                sp_order_no,
                tracking_code,
                name,
                phone,
                email,
                city,
                status,
                created_at,
                picked_up_at,
                delivered_at,
                order_net.get(order_id, 0.0),
            )
        )
    return result


def build_refund_item_totals(
    conn: sqlite3.Connection,
    days: int | None = None,
    start: str | None = None,
    end: str | None = None,
):
    rows = _refund_rows(conn, days, start, end)
    if not rows:
        return {}
    inv_by_number = {}
    inv_by_digits = {}
    for inv_id, inv_no, note, basis in conn.execute(
        "SELECT id, number, note, basis FROM invoices"
    ).fetchall():
        inv_by_number[str(inv_no or "")] = int(inv_id)
        digits = invoice_digits(inv_no)
        if digits:
            inv_by_digits.setdefault(digits, set()).add(int(inv_id))
        for text in (note, basis):
            _, digits_text = extract_invoice_no_from_text(text or "")
            if digits_text:
                inv_by_digits.setdefault(digits_text, set()).add(int(inv_id))
    order_by_invoice = {
        int(inv_id): int(order_id)
        for inv_id, order_id in conn.execute(
            "SELECT invoice_id, order_id FROM invoice_matches"
        ).fetchall()
    }
    storno_map = {
        int(row[0]): int(row[1])
        for row in conn.execute(
            "SELECT storno_invoice_id, original_invoice_id FROM invoice_storno"
        ).fetchall()
    }
    items_by_order = {}
    for oid, sku, qty in conn.execute(
        "SELECT order_id, product_code, qty FROM order_items"
    ).fetchall():
        if not sku:
            continue
        items_by_order.setdefault(int(oid), []).append((str(sku), float(qty or 0)))

    seen_invoices = set()
    totals = {}
    for _, _, _, _, _, inv_no, inv_digits in rows:
        inv_id = None
        if inv_no and str(inv_no) in inv_by_number:
            inv_id = inv_by_number[str(inv_no)]
        elif inv_digits and inv_digits in inv_by_digits and len(inv_by_digits[inv_digits]) == 1:
            inv_id = next(iter(inv_by_digits[inv_digits]))
        if inv_id in storno_map:
            inv_id = storno_map[inv_id]
        if not inv_id or inv_id in seen_invoices:
            continue
        seen_invoices.add(inv_id)
        order_id = order_by_invoice.get(inv_id)
        if not order_id:
            continue
        for sku, qty in items_by_order.get(order_id, []):
            totals[sku] = totals.get(sku, 0.0) + qty
    return totals


def report_refund_items_category(
    conn: sqlite3.Connection,
    category: str,
    days: int | None = None,
    start: str | None = None,
    end: str | None = None,
    return_rows: bool = False,
):
    totals = build_refund_item_totals(conn, days, start, end)
    rows = []
    for sku, qty in totals.items():
        cat = kategorija_za_sifru(str(sku))
        if cat != category:
            continue
        rows.append((sku, qty, cat))
    rows.sort(key=lambda x: x[1], reverse=True)
    print(f"Povrati artikli ({category}): {len(rows)}")
    if return_rows:
        return ["sku", "qty_refund", "category"], rows


def get_needs_invoice_orders(conn: sqlite3.Connection, limit: int = 50):
    rows = conn.execute(
        "SELECT o.id, o.sp_order_no, o.customer_name, o.picked_up_at, o.created_at, o.status, "
        "MAX(oi.cod_amount), MIN(oi.cod_amount), SUM(oi.cod_amount), "
        "MAX(oi.addon_cod), MIN(oi.addon_cod), SUM(oi.addon_cod), "
        "MAX(oi.advance_amount), MIN(oi.advance_amount), SUM(oi.advance_amount), "
        "MAX(oi.addon_advance), MIN(oi.addon_advance), SUM(oi.addon_advance) "
        "FROM order_flags f "
        "JOIN orders o ON o.id = f.order_id "
        "LEFT JOIN order_items oi ON oi.order_id = o.id "
        "WHERE f.flag = 'needs_invoice' "
        "GROUP BY o.id "
        "ORDER BY o.picked_up_at "
        "LIMIT ?",
        (limit,),
    ).fetchall()

    order_ids = [int(row[0]) for row in rows]
    net_map = build_order_net_map(conn, order_ids)

    inv_rows = conn.execute(
        "SELECT i.id, i.turnover, i.amount_due, "
        "CASE WHEN m.id IS NULL THEN 0 ELSE 1 END AS is_matched "
        "FROM invoices i "
        "LEFT JOIN invoice_matches m ON m.invoice_id = i.id"
    ).fetchall()
    matched_map = {
        int(row[0]): str(row[1])
        for row in conn.execute(
            "SELECT m.invoice_id, o.sp_order_no "
            "FROM invoice_matches m "
            "JOIN orders o ON o.id = m.order_id"
        ).fetchall()
    }
    invoices = []
    invoices_by_date = {}
    invoices_no_date = []
    for row in inv_rows:
        inv = {
            "id": int(row[0]),
            "date": normalize_date(row[1]),
            "amount": row[2],
            "matched": bool(row[3]),
        }
        invoices.append(inv)
        if inv["date"]:
            invoices_by_date.setdefault(inv["date"], []).append(inv)
        else:
            invoices_no_date.append(inv)

    def amount_exact(a, b) -> bool:
        return amount_exact_strict(a, b)

    def date_in_window(d1, d2, days_back: int = 10, days_forward: int = 10) -> bool:
        if not d1 or not d2:
            return False
        delta = (d2 - d1).days
        return -days_back <= delta <= days_forward

    results = []
    for row in rows:
        order_id = int(row[0])
        sp_no = row[1]
        name = row[2]
        picked_up_at = row[3]
        created_at = row[4]
        status = row[5]
        display_date = picked_up_at or created_at
        if is_cancelled_status(status) or is_in_progress_status(status):
            reason = "Otkazano"
            results.append((sp_no, name, display_date, reason))
            continue
        amount = net_map.get(order_id)
        order_date = normalize_date(display_date)

        if amount is None or abs(amount) < 0.01:
            continue

        if order_date:
            date_candidates = []
            for offset in range(-10, 11):
                date_candidates.extend(
                    invoices_by_date.get(order_date + timedelta(days=offset), [])
                )
            date_candidates.extend(invoices_no_date)
        else:
            date_candidates = invoices

        amount_candidates = [
            inv for inv in date_candidates if amount_exact(amount, inv.get("amount"))
        ]
        unmatched_candidates = [inv for inv in amount_candidates if not inv.get("matched")]
        matched_candidates = [inv for inv in amount_candidates if inv.get("matched")]

        recent = False
        if order_date:
            try:
                recent = (date.today() - order_date).days <= 3
            except Exception:
                recent = False

        if not order_date:
            if amount_candidates:
                reason = "Nema datuma (ima iznos poklapanje)"
            else:
                reason = "Nema datuma i nema iznos poklapanja"
        elif recent:
            continue
        elif not date_candidates:
            reason = "Nema racuna u -10/+10 dana"
        elif not amount_candidates:
            reason = "Nema racuna sa iznosom u -10/+10 dana"
        elif unmatched_candidates:
            reason = "Ima kandidata po datumu/iznosu (manual)"
        elif matched_candidates:
            other_orders = [
                matched_map.get(inv["id"])
                for inv in matched_candidates
                if matched_map.get(inv["id"])
            ]
            if other_orders:
                reason = f"Racun uparen sa SP: {', '.join(sorted(set(other_orders)))}"
            else:
                reason = "Racun vec uparen"
        else:
            reason = "Nema kandidata"

        results.append((sp_no, name, display_date, reason))

    return results


def get_unmatched_orders_list(conn: sqlite3.Connection, limit: int = 50):
    rows = conn.execute(
        "SELECT o.sp_order_no, o.customer_name, o.picked_up_at "
        "FROM orders o "
        "LEFT JOIN order_items oi ON oi.order_id = o.id "
        "WHERE o.id NOT IN (SELECT order_id FROM invoice_matches) "
        "AND o.id NOT IN (SELECT order_id FROM order_flags WHERE flag = 'needs_invoice') "
        "AND (o.status IS NULL OR lower(o.status) NOT LIKE '%otkazan%') "
        "AND date(substr(COALESCE(o.picked_up_at, o.created_at), 1, 10)) <= date('now', '-3 day') "
        "GROUP BY o.id "
        "HAVING SUM("
        "COALESCE(oi.cod_amount, 0) * (1 - COALESCE(oi.discount, 0) / 100.0) "
        "+ COALESCE(oi.addon_cod, 0) * (1 - COALESCE(oi.extra_discount, 0) / 100.0) "
        "- COALESCE(oi.advance_amount, 0) - COALESCE(oi.addon_advance, 0)"
        ") != 0 "
        "ORDER BY o.picked_up_at "
        "LIMIT ?",
        (limit,),
    ).fetchall()
    return rows


def run_ui(db_path: Path) -> None:
    import customtkinter as ctk
    import concurrent.futures
    import os
    import tempfile
    import threading
    import tkinter as tk
    import time
    from tkinter import filedialog, messagebox
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure

    def acquire_app_lock():
        lock_path = Path(tempfile.gettempdir()) / "srb1_app.lock"
        handle = lock_path.open("a+")
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            handle.close()
            return None
        handle.seek(0)
        handle.truncate()
        handle.write(str(os.getpid()))
        handle.flush()
        return handle

    lock_handle = acquire_app_lock()
    if not lock_handle:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Info", "Aplikacija je vec pokrenuta.")
        root.destroy()
        return

    ctk.set_appearance_mode("light")
    ctk.set_default_color_theme("blue")

    app = ctk.CTk()
    app.title("SRB1.1 - Kontrola i Analitika")
    app.geometry("1200x750")

    def on_close():
        if messagebox.askyesno("Potvrda", "Da li želite završiti rad u aplikaciji?"):
            try:
                lock_handle.close()
            except Exception:
                pass
            app.destroy()

    app.protocol("WM_DELETE_WINDOW", on_close)

    settings = load_app_settings()
    stored_db = settings.get("db_path")
    if stored_db:
        db_path = Path(stored_db)
    state = {
        "db_path": db_path,
        "period_days": None,
        "period_start": None,
        "period_end": None,
        "expense_period_days": None,
        "expense_period_start": None,
        "expense_period_end": None,
        "expense_year": None,
        "expense_month": None,
        "expense_top_n": 5,
        "prodaja_period_days": None,
        "prodaja_period_start": None,
        "prodaja_period_end": None,
        "kalkulacije_output_dir": str(KALKULACIJE_OUT_DIR.resolve()),
        "kartice_output_dir": str(KALKULACIJE_OUT_DIR.resolve()),
    }
    state["kalkulacije_output_dir"] = str(
        Path(settings.get("kalkulacije_output_dir", state["kalkulacije_output_dir"]))
    )
    state["kartice_output_dir"] = str(
        Path(settings.get("kartice_output_dir", state["kartice_output_dir"]))
    )
    btn_reset_matches = None
    def get_conn():
        conn = connect_db(state["db_path"])
        init_db(conn)
        ensure_customer_keys(conn)
        return conn

    def load_currency_mode():
        conn = get_conn()
        try:
            stored = get_app_state(conn, "currency_mode")
        finally:
            conn.close()
        state["currency_mode"] = stored or "RSD"

    def load_baseline_lock():
        conn = get_conn()
        try:
            locked = get_app_state(conn, "baseline_locked")
            locked_at = get_app_state(conn, "baseline_locked_at")
        finally:
            conn.close()
        state["baseline_locked"] = locked == "1"
        state["baseline_locked_at"] = locked_at

    def refresh_exchange_rate():
        conn = get_conn()
        error = None
        try:
            rate, error = fetch_cbbh_rsd_rate_debug()
            if rate is not None:
                set_app_state(conn, "rate_rsd_to_bam", f"{rate}")
            else:
                stored = get_app_state(conn, "rate_rsd_to_bam")
                rate = float(stored) if stored else None
            state["rate_rsd_to_bam"] = rate
        finally:
            conn.close()
        return rate, error

    def format_amount(amount: float) -> str:
        try:
            rsd_val = float(amount)
        except (TypeError, ValueError):
            return "0.00 RSD"
        rate = state.get("rate_rsd_to_bam")
        mode = state.get("currency_mode", "RSD")
        def fmt(val: float) -> str:
            return f"{val:,.0f}".replace(",", ".")
        if mode == "BAM":
            if rate is None:
                return f"{fmt(rsd_val)} RSD"
            bam_val = rsd_val * rate
            return f"{fmt(bam_val)} BAM"
        if rate is None:
            return f"{fmt(rsd_val)} RSD"
        bam_val = rsd_val * rate
        return f"{fmt(rsd_val)} RSD ({fmt(bam_val)} BAM)"

    def format_amount_rsd(amount: float) -> str:
        try:
            rsd_val = float(amount)
        except (TypeError, ValueError):
            return "0.00 RSD"
        return f"{rsd_val:,.2f}".replace(",", ".") + " RSD"

    def chart_currency_label() -> str:
        if state.get("currency_mode") == "BAM" and state.get("rate_rsd_to_bam") is not None:
            return "BAM"
        return "RSD"

    def chart_value(value) -> float:
        try:
            val = float(value or 0)
        except (TypeError, ValueError):
            return 0.0
        if state.get("currency_mode") == "BAM":
            rate = state.get("rate_rsd_to_bam")
            if rate is not None:
                return val * rate
        return val

    def _parse_user_date(text: str) -> date | None:
        value = (text or "").strip()
        if not value:
            return None
        for fmt in ("%Y-%m-%d", "%d.%m.%Y"):
            try:
                return datetime.strptime(value, fmt).date()
            except ValueError:
                continue
        return None

    def _load_prodaja_dataframe() -> pd.DataFrame:
        conn = get_conn()
        try:
            rows = conn.execute(
                "SELECT sku, artikal, total_sold, first_date, last_date, total_days, "
                "days_without, days_available, avg_daily, total_net, avg_net, lost_net, lost_qty "
                "FROM prodaja_stats"
            ).fetchall()
        finally:
            conn.close()
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(
            rows,
            columns=[
                "SKU",
                "Artikal",
                "Total prodato",
                "Prvi datum",
                "Zadnji datum",
                "Ukupno dana",
                "Dani bez zalihe",
                "Dani dostupno",
                "Prosek dnevno",
                "Ukupno neto",
                "Prosek neto",
                "Procjena gubitka neto",
                "Procena izgubljeno",
            ],
        )
        for col in ("Prvi datum", "Zadnji datum"):
            df[col] = pd.to_datetime(df[col], errors="coerce")
        return df

    def _resolve_prodaja_period():
        start = state.get("prodaja_period_start")
        end = state.get("prodaja_period_end")
        days = state.get("prodaja_period_days")
        if days and start is None and end is None:
            today = datetime.utcnow().date()
            end = today
            start = today - timedelta(days=days - 1)
        return start, end

    def _filter_prodaja_by_period(
        df: pd.DataFrame, start: date | None, end: date | None
    ) -> pd.DataFrame:
        if df.empty:
            return df
        mask = pd.Series(True, index=df.index)
        if start is not None and "Zadnji datum" in df.columns:
            mask &= df["Zadnji datum"].dt.date >= start
        if end is not None and "Prvi datum" in df.columns:
            mask &= df["Prvi datum"].dt.date <= end
        return df.loc[mask]

    def update_cartice_ui():
        parts: list[str] = []
        conn = get_conn()
        try:
            for source in (ZERO_SOURCE, KALKULACIJE_SOURCE, PRODAJA_SOURCE):
                info = _get_last_import_info(conn, source)
                label = SOURCE_LABELS.get(source, source)
                if info:
                    imported_at, row_count = info
                    parts.append(f"{label}: {imported_at} ({row_count} redova)")
                else:
                    parts.append(f"{label}: nema uvoza")
        finally:
            conn.close()
        cartice_status_var.set(" | ".join(parts))

    def _show_import_summary(title: str, summary: list[dict[str, str | int]]) -> None:
        lines: list[str] = []
        for result in summary:
            label = SOURCE_LABELS.get(result["source"], result["source"])
            status = result.get("status")
            if status == "missing":
                lines.append(f"{label}: fajl nije pronaden ({result.get('path')})")
            elif status == "skipped":
                lines.append(f"{label}: podaci vec postoje")
            elif status == "imported":
                rows = result.get("rows", 0)
                lines.append(f"{label}: uvezeno {rows} redova")
        message = "\n".join(lines) if lines else "Nema novih podataka za import."
        messagebox.showinfo(title, message)

    def _run_cartice_import_batch(
        title: str, batch: list[tuple[Path, str, callable]]
    ) -> list[dict[str, str | int]]:
        if state.get("baseline_locked"):
            messagebox.showwarning(
                "Baza zakljucana",
                "Trenutno stanje je zakljucano kao pocetno, resetuj bazu da bi uvezao nove podatke.",
            )
            return []
        conn = get_conn()
        summary: list[dict[str, str | int]] = []
        try:
            for path, source, fn in batch:
                summary.append(_run_cartice_import(conn, path, source, fn))
        except Exception as exc:
            messagebox.showerror(title, f"Greska pri importu: {exc}")
            return []
        finally:
            conn.close()
        refresh_prodaja_tab()
        update_cartice_ui()
        _show_import_summary(title, summary)
        return summary

    def import_cartice_data():
        _run_cartice_import_batch(
            "Import kartice",
            [
                (KALKULACIJE_DETAIL_CSV, KALKULACIJE_DETAIL_SOURCE, _apply_kalkulacije_detail),
                (ZERO_INTERVAL_CSV, ZERO_SOURCE, _apply_zero_intervals),
                (KALKULACIJE_AGG_CSV, KALKULACIJE_SOURCE, _apply_kalkulacije),
                (PRODAJA_STATS_CSV, PRODAJA_SOURCE, _apply_prodaja_stats),
            ],
        )

    def _run_external_parser(label: str, cmd: list[str]) -> bool:
        status_var.set(f"{label}: pokrećem parser...")
        app.update_idletasks()
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:
            error = (exc.stderr or exc.stdout or str(exc)).strip()
            log_app_error(label, error)
            messagebox.showerror("Parser greška", f"{label}:\n{error}")
            status_var.set("Greska.")
            return False
        except FileNotFoundError as exc:
            log_app_error(label, str(exc))
            messagebox.showerror("Parser greška", str(exc))
            status_var.set("Greska.")
            return False
        status_var.set("Spremno.")
        return True

    def _choose_output_dir(state_key: str, var: ctk.StringVar, title: str) -> None:
        folder = filedialog.askdirectory(
            title=title, initialdir=state.get(state_key) or str(KALKULACIJE_OUT_DIR)
        )
        if not folder:
            return
        resolved = str(Path(folder).resolve())
        state[state_key] = resolved
        var.set(resolved)
        save_app_settings({state_key: resolved})

    def import_kalkulacije_flow() -> None:
        if state.get("baseline_locked"):
            messagebox.showwarning(
                "Baza zakljucana",
                "Trenutno stanje je zakljucano kao pocetno, resetuj bazu da bi uvezao nove podatke.",
            )
            return
        excel_path = filedialog.askopenfilename(
            title="Odaberi Excel kalkulaciju",
            initialdir=str(KALKULACIJE_DIR),
            filetypes=[("Excel fajl", "*.xlsx"), ("Svi fajlovi", "*.*")],
        )
        if not excel_path:
            return
        output_dir = Path(state["kalkulacije_output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            str(EXTRACT_SCRIPT),
            "--excel",
            excel_path,
            "--out",
            str(output_dir),
            "--skip-pdf",
        ]
        if not _run_external_parser("Kalkulacije", cmd):
            return
        _run_cartice_import_batch(
            "Import kalkulacija",
            [
                (output_dir / "kalkulacije_marza.csv", KALKULACIJE_DETAIL_SOURCE, _apply_kalkulacije_detail),
                (output_dir / "kalkulacije_marza_agregat.csv", KALKULACIJE_SOURCE, _apply_kalkulacije),
                (output_dir / "prodaja_avg_i_gubitak.csv", PRODAJA_SOURCE, _apply_prodaja_stats),
            ],
        )

    def import_kartice_flow() -> None:
        if state.get("baseline_locked"):
            messagebox.showwarning(
                "Baza zakljucana",
                "Trenutno stanje je zakljucano kao pocetno, resetuj bazu da bi uvezao nove podatke.",
            )
            return
        pdf_path = filedialog.askopenfilename(
            title="Odaberi PDF karticu artikala",
            initialdir=str(KALKULACIJE_DIR),
            filetypes=[("PDF fajl", "*.pdf"), ("Svi fajlovi", "*.*")],
        )
        if not pdf_path:
            return
        output_dir = Path(state["kartice_output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            str(EXTRACT_SCRIPT),
            "--pdf",
            pdf_path,
            "--out",
            str(output_dir),
            "--skip-excel",
        ]
        if not _run_external_parser("Kartice artikala", cmd):
            return
        _run_cartice_import_batch(
            "Import kartice artikala",
            [
                (output_dir / "kartice_zero_intervali.csv", ZERO_SOURCE, _apply_zero_intervals),
                (output_dir / "prodaja_avg_i_gubitak.csv", PRODAJA_SOURCE, _apply_prodaja_stats),
            ],
        )

    def show_latest_kalkulacija():
        conn = get_conn()
        try:
            row = conn.execute(
                "SELECT opis, sku, artikal, datum, broj, imported_at "
                "FROM cartice_kalkulacije_detail "
                "ORDER BY imported_at DESC LIMIT 1"
            ).fetchone()
        finally:
            conn.close()
        if not row:
            messagebox.showinfo("Najnovija kalkulacija", "Ne postoji nijedna uvezena kalkulacija.")
            return
        opis, sku, artikal, datum, broj, imported_at = row
        info = (
            f"Opis: {opis}\n"
            f"SKU: {sku or '-'}\n"
            f"Artikal: {artikal or '-'}\n"
            f"Datum kalkulacije: {datum or '-'}\n"
            f"Broj: {broj or '-'}\n"
            f"Uvezeno: {imported_at or '-'}"
        )
        messagebox.showinfo("Najnovija kalkulacija", info)

    def get_progress_info(task: str):
        conn = get_conn()
        try:
            row = get_task_progress(conn, task)
        finally:
            conn.close()
        return row

    def poll_global_status():
        conn = get_conn()
        try:
            latest = get_latest_task_progress(conn)
        finally:
            conn.close()
        if latest:
            total = latest.get("total", 0)
            processed = latest.get("processed", 0)
            task = latest.get("task", "-")
            if total:
                pct = int((processed / total) * 100)
                task_status_var.set(f"Task: {task} {processed}/{total} ({pct}%)")
            else:
                task_status_var.set(f"Task: {task}")
        else:
            task_status_var.set("Task: idle")
        log_path = Path("exports") / "app-errors.log"
        if log_path.exists():
            try:
                last_line = log_path.read_text(encoding="utf-8").strip().splitlines()[-1]
                last_error_var.set(f"Zadnja greska: {last_line[:120]}")
            except Exception:
                last_error_var.set("Zadnja greska: -")
        else:
            last_error_var.set("Zadnja greska: -")
        app.after(3000, poll_global_status)

    def format_eta(seconds: float | None) -> str:
        if seconds is None or seconds < 0:
            return "--"
        mins, secs = divmod(int(seconds), 60)
        if mins >= 60:
            hrs, mins = divmod(mins, 60)
            return f"{hrs}h {mins}m"
        if mins > 0:
            return f"{mins}m {secs}s"
        return f"{secs}s"

    def refresh_dashboard():
        conn = get_conn()
        kpis = get_kpis(conn, state.get("period_days"), state.get("period_start"), state.get("period_end"))
        unpicked_all = conn.execute(
            "SELECT COUNT(*) FROM orders "
            "WHERE status LIKE '%Vra\u0107eno%' OR status LIKE '%Vraceno%'"
        ).fetchone()[0]
        lbl_total_orders.configure(text=str(kpis["total_orders"]))
        lbl_total_revenue.configure(text=format_amount(kpis["total_revenue"]))
        lbl_returns.configure(text=str(unpicked_all or 0))
        lbl_unmatched.configure(text=str(kpis["unmatched"]))
        conn.close()
        refresh_charts()
        refresh_expenses()
        refresh_returns_charts()
        refresh_unpicked_charts()
        refresh_prodaja_tab()
        try:
            status_var.set("Osvjezeno.")
            app.after(1500, lambda: status_var.set("Spremno."))
        except Exception:
            pass

    def _truncate_label(text: str, max_chars: int) -> str:
        if max_chars <= 0:
            return ""
        if len(text) <= max_chars:
            return text
        if max_chars <= 3:
            return text[:max_chars]
        return text[: max_chars - 3] + "..."

    def _fit_two_line_label(name: str, count_label: str, value_text: str, ratio: float) -> str:
        max_line1 = int(10 + ratio * 24)
        max_line2 = int(12 + ratio * 22)
        line1 = _truncate_label(name, max_line1)
        line2 = _truncate_label(f"{count_label} | {value_text}", max_line2)
        return f"{line1}\n{line2}"

    def refresh_charts():
        conn = get_conn()
        period_days = state.get("period_days")
        start = state.get("period_start")
        end = state.get("period_end")
        top_customers = get_top_customers(conn, 5, period_days, start, end)
        top_products = get_top_products(conn, 10, period_days, start, end)
        monthly = get_sp_bank_monthly(conn, period_days, start, end)
        conn.close()

        ax_customers.clear()
        if top_customers:
            values = [chart_value(r[2]) for r in top_customers]
            max_val = max(values) if values else 0
            labels = []
            for name, orders_cnt, net_total in top_customers:
                ratio = (chart_value(net_total) / max_val) if max_val else 0
                labels.append(
                    _fit_two_line_label(
                        str(name),
                        f"N: {int(orders_cnt or 0)}",
                        format_amount(net_total),
                        ratio,
                    )
                )
            labels_rev = labels[::-1]
            values_rev = values[::-1]
            y_pos = list(range(len(labels_rev)))
            bars = ax_customers.barh(y_pos, values_rev, color="#5aa9e6")
            ax_customers.set_title(f"Top 5 kupaca ({chart_currency_label()})")
            ax_customers.tick_params(axis="y", left=False, labelleft=False)
            outside_labels = []
            outside_positions = []
            inside_labels = []
            for bar, label in zip(bars, labels_rev):
                width = bar.get_width()
                ratio = (width / max_val) if max_val else 0
                if ratio < 0.22:
                    outside_labels.append(label)
                    outside_positions.append(bar)
                    inside_labels.append("")
                else:
                    inside_labels.append(label)
            ax_customers.bar_label(
                bars,
                labels=inside_labels,
                label_type="center",
                padding=0,
                color="white",
                fontsize=8,
            )
            if outside_positions:
                offset = max_val * 0.02 if max_val else 0.1
                for bar, label in zip(outside_positions, outside_labels):
                    ax_customers.text(
                        bar.get_width() + offset,
                        bar.get_y() + bar.get_height() / 2,
                        label,
                        va="center",
                        ha="left",
                        fontsize=8,
                        color="black",
                    )
        else:
            ax_customers.set_title("Top 5 kupaca (nema podataka)")
        canvas_customers.draw()

        ax_products.clear()
        if top_products:
            names = [f"{r[0]} ({int(r[1] or 0)})" for r in top_products]
            values = [chart_value(r[2]) for r in top_products]
            names_rev = names[::-1]
            values_rev = values[::-1]
            y_pos = list(range(len(names_rev)))
            bars = ax_products.barh(y_pos, values_rev, color="#7cb518")
            ax_products.set_title(f"Top 10 artikala ({chart_currency_label()})")
            ax_products.tick_params(axis="y", left=False, labelleft=False)
            ax_products.bar_label(
                bars,
                labels=names_rev,
                label_type="center",
                padding=0,
                color="white",
                fontsize=9,
            )
        else:
            ax_products.set_title("Top 10 artikala (nema podataka)")
        canvas_products.draw()

        ax_sp_bank.clear()
        if monthly:
            periods = [r[0] for r in monthly]
            income = [r[1] or 0 for r in monthly]
            expense = [r[2] or 0 for r in monthly]
            x = range(len(periods))
            ax_sp_bank.bar(x, income, width=0.4, label="Prihodi", color="#5aa9e6")
            ax_sp_bank.bar([i + 0.4 for i in x], expense, width=0.4, label="Rashodi", color="#f28482")
            ax_sp_bank.set_xticks(list(x))
            ax_sp_bank.set_xticklabels(periods, rotation=45, ha="right")
            ax_sp_bank.set_title("Prihodi vs rashodi (banka, BAM)")
            ax_sp_bank.legend()
        else:
            ax_sp_bank.set_title("Prihodi vs rashodi (nema podataka)")
        canvas_sp_bank.draw()

    def refresh_expenses():
        conn = get_conn()
        years = [
            row[0]
            for row in conn.execute(
                "SELECT DISTINCT substr(dtposted, 1, 4) "
                "FROM bank_transactions "
                "WHERE dtposted IS NOT NULL "
                "ORDER BY substr(dtposted, 1, 4)"
            ).fetchall()
            if row[0]
        ]
        if years:
            expense_year_menu.configure(values=["Sve"] + years)
            if expense_year_var.get() not in (["Sve"] + years):
                expense_year_var.set("Sve")
                state["expense_year"] = None
        period_days = state.get("expense_period_days")
        start = state.get("expense_period_start")
        end = state.get("expense_period_end")
        year = state.get("expense_year")
        month = state.get("expense_month")
        if year or month:
            period_days = None
            start = None
            end = None
        summary = get_expense_summary(conn, period_days, start, end, year, month)
        conn.close()

        total = summary["total"] or 0.0
        totals = summary["totals"]
        display = summary["display_names"]
        monthly = summary["monthly"]
        expense_total_var.set(f"Ukupno: {format_amount(total)}")

        top_n = state.get("expense_top_n", 5)
        top_items = sorted(totals.items(), key=lambda item: item[1], reverse=True)[:top_n]

        ax_expenses_top.clear()
        if top_items:
            values = [chart_value(v) for _, v in top_items]
            max_val = max(values) if values else 0
            labels = []
            full_labels = []
            for key, val in top_items:
                name = display.get(key, key)
                share = (val / total * 100.0) if total else 0.0
                ratio = (chart_value(val) / max_val) if max_val else 0
                label = _fit_two_line_label(
                    name,
                    f"Udio: {share:.1f}%",
                    format_amount(val),
                    ratio,
                )
                labels.append(label)
                full_labels.append(f"{name}\nUdio: {share:.1f}% | {format_amount(val)}")
            labels_rev = labels[::-1]
            full_labels_rev = full_labels[::-1]
            values_rev = values[::-1]
            y_pos = list(range(len(labels_rev)))
            bars = ax_expenses_top.barh(y_pos, values_rev, color="#f28e2c")
            ax_expenses_top.set_title(f"Top {len(top_items)} troskova ({chart_currency_label()})")
            ax_expenses_top.tick_params(axis="y", left=False, labelleft=False)
            outside_labels = []
            outside_positions = []
            inside_labels = []
            for bar, label, full_label in zip(bars, labels_rev, full_labels_rev):
                width = bar.get_width()
                ratio = (width / max_val) if max_val else 0
                if ratio < 0.22:
                    outside_labels.append(full_label)
                    outside_positions.append(bar)
                    inside_labels.append("")
                else:
                    inside_labels.append(label)
            ax_expenses_top.bar_label(
                bars,
                labels=inside_labels,
                label_type="center",
                padding=0,
                color="white",
                fontsize=8,
            )
            if outside_positions:
                offset = max_val * 0.02 if max_val else 0.1
                for bar, label in zip(outside_positions, outside_labels):
                    ax_expenses_top.text(
                        bar.get_width() + offset,
                        bar.get_y() + bar.get_height() / 2,
                        label,
                        va="center",
                        ha="left",
                        fontsize=8,
                        color="black",
                    )
                ax_expenses_top.set_xlim(0, max_val * 1.25 if max_val else 1)
        else:
            ax_expenses_top.set_title("Top troskovi (nema podataka)")
        canvas_expenses_top.draw()

        ax_expenses_month.clear()
        if monthly and top_items:
            months = sorted(monthly.keys())
            top_keys = [k for k, _ in top_items]
            bottoms = [0.0 for _ in months]
            for key in top_keys:
                series = [chart_value(monthly.get(m, {}).get(key, 0.0)) for m in months]
                total_val = totals.get(key, 0.0)
                share = (total_val / total * 100.0) if total else 0.0
                name = _short_expense_name(display.get(key, key))
                label = f"{name} | {format_amount(total_val)} | {share:.1f}%"
                ax_expenses_month.bar(months, series, bottom=bottoms, label=label)
                bottoms = [b + s for b, s in zip(bottoms, series)]
            ax_expenses_month.set_title(f"Troskovi po mjesecu (Top {len(top_items)})")
            ax_expenses_month.tick_params(axis="x", rotation=35)
            ax_expenses_month.legend(fontsize=7)
        else:
            ax_expenses_month.set_title("Troskovi po mjesecu (nema podataka)")
        canvas_expenses_month.draw()

    def refresh_returns_charts():
        conn = get_conn()
        period_days = state.get("period_days")
        start = state.get("period_start")
        end = state.get("period_end")
        top_customers = get_refund_top_customers(conn, 5, period_days, start, end)
        top_items = get_refund_top_items(conn, 5, period_days, start, end)
        top_categories = get_refund_top_categories(conn, 5, period_days, start, end)
        conn.close()

        ax_ref_customers.clear()
        if top_customers:
            values = [r[1] or 0 for r in top_customers]
            max_val = max(values) if values else 0
            labels = []
            for name, count, total_amount in top_customers:
                ratio = (float(count or 0) / max_val) if max_val else 0
                labels.append(
                    _fit_two_line_label(
                        str(name),
                        f"P: {int(count or 0)}",
                        format_amount(total_amount or 0),
                        ratio,
                    )
                )
            labels_rev = labels[::-1]
            values_rev = values[::-1]
            y_pos = list(range(len(labels_rev)))
            bars = ax_ref_customers.barh(y_pos, values_rev, color="#3a86ff")
            ax_ref_customers.set_title("Top 5 kupaca (povrati, broj)")
            ax_ref_customers.tick_params(axis="y", left=False, labelleft=False)
            ax_ref_customers.bar_label(
                bars,
                labels=labels_rev,
                label_type="center",
                padding=0,
                color="white",
                fontsize=9,
            )
        else:
            ax_ref_customers.set_title("Top 5 kupaca (povrati, nema podataka)")
        canvas_ref_customers.draw()

        ax_ref_items.clear()
        if top_items:
            names = [f"{r[0]} ({int(r[1] or 0)})" for r in top_items]
            values = [r[1] or 0 for r in top_items]
            names_rev = names[::-1]
            values_rev = values[::-1]
            y_pos = list(range(len(names_rev)))
            bars = ax_ref_items.barh(y_pos, values_rev, color="#2a9d8f")
            ax_ref_items.set_title("Top 5 artikala (povrati, kom)")
            ax_ref_items.tick_params(axis="y", left=False, labelleft=False)
            ax_ref_items.bar_label(
                bars,
                labels=names_rev,
                label_type="center",
                padding=0,
                color="white",
                fontsize=9,
            )
        else:
            ax_ref_items.set_title("Top 5 artikala (povrati, nema podataka)")
        canvas_ref_items.draw()

        ax_ref_categories.clear()
        if top_categories:
            names = [f"{r[0]} ({int(r[1] or 0)})" for r in top_categories]
            values = [r[1] or 0 for r in top_categories]
            names_rev = names[::-1]
            values_rev = values[::-1]
            y_pos = list(range(len(names_rev)))
            bars = ax_ref_categories.barh(y_pos, values_rev, color="#ff9f1c")
            ax_ref_categories.set_title("Top 5 grupa (povrati, kom)")
            ax_ref_categories.tick_params(axis="y", left=False, labelleft=False)
            ax_ref_categories.bar_label(
                bars,
                labels=names_rev,
                label_type="center",
                padding=0,
                color="white",
                fontsize=9,
            )
        else:
            ax_ref_categories.set_title("Top 5 grupa (povrati, nema podataka)")
        canvas_ref_categories.draw()

    def refresh_unpicked_charts():
        conn = get_conn()
        period_days = state.get("unpicked_period_days")
        start = state.get("unpicked_period_start")
        end = state.get("unpicked_period_end")
        stats = get_unpicked_stats(conn, period_days, start, end)
        top_customers, _ = get_unpicked_customer_groups(conn, 10, period_days, start, end)
        top_items = get_unpicked_top_items(conn, 5, period_days, start, end)
        orders = get_unpicked_orders_list(conn, period_days, start, end)
        tracking_codes = [row[1] for row in orders if row[1]]
        summary_map = {}
        if tracking_codes:
            placeholders = ",".join("?" * len(tracking_codes))
            for row in conn.execute(
                "SELECT tracking_code, delivery_attempts, failure_reasons, returned_at, "
                "days_to_first_attempt, has_attempt_before_return, has_returned, anomalies "
                f"FROM tracking_summary WHERE tracking_code IN ({placeholders})",
                tracking_codes,
            ).fetchall():
                summary_map[row[0]] = row[1:]
        conn.close()

        lbl_unpicked_total.configure(text=f"Nepreuzete: {stats['unpicked_orders']}")
        lbl_unpicked_lost.configure(text=f"Izgubljena prodaja: {format_amount(stats['lost_sales'])}")
        lbl_unpicked_repeat.configure(text=f"Kupci 2+ nepreuzetih: {stats['repeat_customers']}")

        ax_unpicked_customers.clear()
        if top_customers:
            names = [f"{r[0]} ({int(r[1] or 0)})" for r in top_customers]
            values = [r[1] or 0 for r in top_customers]
            names_rev = names[::-1]
            values_rev = values[::-1]
            y_pos = list(range(len(names_rev)))
            bars = ax_unpicked_customers.barh(y_pos, values_rev, color="#ff7f50")
            ax_unpicked_customers.set_title("Top 10 kupaca (nepreuzete, broj)")
            ax_unpicked_customers.tick_params(axis="y", left=False, labelleft=False)
            ax_unpicked_customers.bar_label(
                bars,
                labels=names_rev,
                label_type="center",
                padding=0,
                color="white",
                fontsize=9,
            )
        else:
            ax_unpicked_customers.set_title("Top 10 kupaca (nepreuzete, nema podataka)")
        canvas_unpicked_customers.draw()

        ax_unpicked_items.clear()
        if top_items:
            names = [f"{r[0]} ({int(r[1] or 0)})" for r in top_items]
            values = [r[1] or 0 for r in top_items]
            names_rev = names[::-1]
            values_rev = values[::-1]
            y_pos = list(range(len(names_rev)))
            bars = ax_unpicked_items.barh(y_pos, values_rev, color="#2a9d8f")
            ax_unpicked_items.set_title("Top 5 artikala (nepreuzete, kom)")
            ax_unpicked_items.tick_params(axis="y", left=False, labelleft=False)
            ax_unpicked_items.bar_label(
                bars,
                labels=names_rev,
                label_type="center",
                padding=0,
                color="white",
                fontsize=9,
            )
        else:
            ax_unpicked_items.set_title("Top 5 artikala (nepreuzete, nema podataka)")
        canvas_unpicked_items.draw()

        attempts_count = {"0": 0, "1": 0, "2": 0, "3+": 0}
        reasons_count = {}
        anomalies_count = {}
        days_to_first = []
        out_to_return = []
        no_attempt_before_return = 0
        not_returned = 0
        one_attempt = 0
        late_first_attempt = 0

        for code, summary in summary_map.items():
            attempts, reasons, returned_at, days_first, has_attempt_before_return, has_returned, anomalies = summary
            attempts = int(attempts or 0)
            if has_returned == 0:
                not_returned += 1
            if has_returned == 1 and has_attempt_before_return == 0:
                no_attempt_before_return += 1
            if attempts == 1:
                one_attempt += 1
            if attempts >= 3:
                attempts_count["3+"] += 1
            elif attempts == 2:
                attempts_count["2"] += 1
            elif attempts == 1:
                attempts_count["1"] += 1
            else:
                attempts_count["0"] += 1
            if days_first is not None:
                try:
                    val = float(days_first)
                    days_to_first.append(val)
                    if val > 2:
                        late_first_attempt += 1
                except (TypeError, ValueError):
                    pass
            if returned_at and has_attempt_before_return:
                pass
            for reason in [r.strip() for r in (reasons or "").split(";") if r.strip()]:
                reasons_count[reason] = reasons_count.get(reason, 0) + 1
            for anomaly in [a.strip() for a in (anomalies or "").split(";") if a.strip()]:
                anomalies_count[anomaly] = anomalies_count.get(anomaly, 0) + 1

        avg_days_first = sum(days_to_first) / len(days_to_first) if days_to_first else None
        med_days_first = statistics.median(days_to_first) if days_to_first else None
        top_reason = max(reasons_count.items(), key=lambda x: x[1])[0] if reasons_count else ""
        top_anomaly = max(anomalies_count.items(), key=lambda x: x[1])[0] if anomalies_count else ""

        txt_tracking_summary.delete("1.0", "end")
        txt_tracking_summary.insert("end", f"Najcesci razlog: {top_reason or '-'}\n")
        txt_tracking_summary.insert("end", f"Najcesci nelogicni slijed: {top_anomaly or '-'}\n")
        txt_tracking_summary.insert("end", f"Bez pokusaja prije vracanja: {no_attempt_before_return}\n")
        txt_tracking_summary.insert("end", f"Samo jedan pokusaj: {one_attempt}\n")
        txt_tracking_summary.insert("end", f"Prvi pokusaj >2 dana: {late_first_attempt}\n")
        txt_tracking_summary.insert(
            "end",
            f"Prosjek/medijan dana do prvog pokusaja: "
            f"{avg_days_first:.2f}/{med_days_first:.2f}\n" if avg_days_first is not None and med_days_first is not None else
            "Prosjek/medijan dana do prvog pokusaja: -\n",
        )
        txt_tracking_summary.insert("end", f"Distribucija pokusaja: {attempts_count}\n")
        txt_tracking_summary.insert("end", f"Nisu vracene: {not_returned}\n")
        month_counts = {}
        for row in orders:
            created_at = row[7]
            if not created_at:
                continue
            ts = pd.to_datetime(created_at, errors="coerce", dayfirst="." in str(created_at) and "-" not in str(created_at))
            if pd.isna(ts):
                continue
            key = ts.strftime("%Y-%m")
            month_counts[key] = month_counts.get(key, 0) + 1
        if month_counts:
            trend = ", ".join(f"{k}:{v}" for k, v in sorted(month_counts.items())[-6:])
            txt_tracking_summary.insert("end", f"Trend (zadnjih 6): {trend}\n")

        txt_nepreuzete_orders.delete("1.0", "end")
        if orders:
            for row in orders:
                sp_order_no, tracking, name, phone, email, city, status, created_at, picked_up_at, delivered_at, net_total = row
                txt_nepreuzete_orders.insert(
                    "end",
                    f"{sp_order_no} | {tracking or '-'} | {name or ''} | {phone or ''} | {city or ''} | {format_amount(net_total)}\n",
                )
        else:
            txt_nepreuzete_orders.insert("end", "Nema podataka.\n")

    def refresh_poslovanje_lists():
        conn = get_conn()
        needs = get_needs_invoice_orders(conn, 100)
        unmatched = get_unmatched_orders_list(conn, 100)
        conn.close()
        txt_needs.configure(state="normal")
        txt_needs.delete("1.0", "end")
        for idx, r in enumerate(needs, start=1):
            txt_needs.insert("end", f"{idx}. {r[0]} | {r[1]} | {r[2]} | {r[3]}\n")
        txt_needs.configure(state="disabled")

        txt_unmatched.configure(state="normal")
        txt_unmatched.delete("1.0", "end")
        for idx, r in enumerate(unmatched, start=1):
            txt_unmatched.insert("end", f"{idx}. {r[0]} | {r[1]} | {r[2]}\n")
        txt_unmatched.configure(state="disabled")

    def run_export_basic():
        conn = get_conn()
        try:
            cols, rows = report_unmatched_orders(conn, return_rows=True)
            write_report(cols, rows, Path("exports"), "unmatched", "xlsx")
            cols, rows = report_unmatched_reasons(conn, return_rows=True)
            write_report(cols, rows, Path("exports"), "unmatched-reasons", "xlsx")
            cols, rows = report_conflicts(conn, return_rows=True)
            write_report(cols, rows, Path("exports"), "conflicts", "xlsx")
            cols, rows = report_nearest_invoice(conn, return_rows=True)
            write_report(cols, rows, Path("exports"), "nearest", "xlsx")
            cols, rows = report_needs_invoice_orders(conn, return_rows=True)
            write_report(cols, rows, Path("exports"), "needs-invoice-orders", "xlsx")
            cols, rows = report_sp_vs_bank(conn, return_rows=True)
            write_report(cols, rows, Path("exports"), "sp-vs-bank", "xlsx")
            cols, rows = report_refunds_without_storno(conn, return_rows=True)
            write_report(cols, rows, Path("exports"), "refunds-no-storno", "xlsx")
            messagebox.showinfo("OK", "Export zavrsen u folderu exports.")
        except Exception as exc:
            messagebox.showerror("Greska", str(exc))
        finally:
            conn.close()

    def run_export_single(report_fn, name: str):
        conn = get_conn()
        try:
            cols, rows = report_fn(conn, return_rows=True)
            out_path = write_report(cols, rows, Path("exports"), name, "xlsx")
            messagebox.showinfo("OK", f"Export snimljen: {out_path}")
        except Exception as exc:
            messagebox.showerror("Greska", str(exc))
        finally:
            conn.close()

    def open_exports():
        folder = Path("exports").resolve()
        folder.mkdir(parents=True, exist_ok=True)
        try:
            os.startfile(str(folder))
        except Exception as exc:
            messagebox.showerror("Greska", str(exc))

    def set_db_path():
        path = filedialog.askopenfilename(
            title="Izaberi SQLite bazu",
            filetypes=[("SQLite DB", "*.db"), ("All files", "*.*")],
        )
        if not path:
            return
        state["db_path"] = Path(path)
        save_app_settings({"db_path": path})
        ent_db.configure(state="normal")
        ent_db.delete(0, "end")
        ent_db.insert(0, path)
        ent_db.configure(state="readonly")
        refresh_dashboard()

    def run_import_folder(import_fn, title: str, pattern: str):
        folder = filedialog.askdirectory(title=title)
        if not folder:
            return
        files = sorted(Path(folder).glob(pattern))
        if not files:
            messagebox.showwarning("Info", f"Nema fajlova za import ({pattern}).")
            return
        progress.configure(mode="determinate")
        progress.set(0)
        progress_pct_var.set("Napredak: 0%")
        conn = get_conn()
        imported = 0
        skipped = []
        rejects = []
        try:
            total = len(files)
            for idx, path in enumerate(files, start=1):
                digest = file_hash(path)
                exists = conn.execute(
                    "SELECT 1 FROM import_runs WHERE file_hash = ?",
                    (digest,),
                ).fetchone()
                if exists:
                    skipped.append(path.name)
                    append_reject(rejects, title, path.name, None, "file_already_imported", "")
                else:
                    import_fn(conn, path, rejects)
                    imported += 1
                pct = idx / total if total else 1
                progress.set(pct)
                progress_pct_var.set(f"Napredak: {int(pct * 100)}% ({idx}/{total})")
                status_var.set(f"Uvoz: {idx}/{total}")
                app.update_idletasks()
            msg = f"Import zavrsen. Fajlova: {imported}."
            if skipped:
                preview = ", ".join(skipped[:8])
                more = f" (+{len(skipped) - 8})" if len(skipped) > 8 else ""
                msg += f"\nPreskoceno (vec u bazi): {preview}{more}"
            messagebox.showinfo("OK", msg)
            if rejects:
                ts = datetime.now().strftime("%Y%m%d%H%M%S")
                out_path = Path("exports") / f"rejected-rows-{ts}.xlsx"
                pd.DataFrame(rejects).to_excel(out_path, index=False)
                try:
                    os.startfile(out_path)  # type: ignore[attr-defined]
                except Exception:
                    messagebox.showinfo("Info", f"Log odbijenih redova: {out_path}")
        except Exception as exc:
            messagebox.showerror("Greska", str(exc))
        finally:
            conn.close()
            status_var.set("Spremno.")
        refresh_dashboard()

    def show_last_imports():
        conn = get_conn()
        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT MAX(CAST(sp_order_no AS INTEGER)) "
                "FROM orders WHERE sp_order_no IS NOT NULL AND TRIM(sp_order_no) != ''"
            )
            max_sp_orders = cur.fetchone()[0]
            cur.execute(
                "SELECT MAX(CAST(sp_order_no AS INTEGER)) "
                "FROM payments WHERE sp_order_no IS NOT NULL AND TRIM(sp_order_no) != ''"
            )
            max_sp_payments = cur.fetchone()[0]
            cur.execute(
                "SELECT MAX(CAST(sp_order_no AS INTEGER)) "
                "FROM returns WHERE sp_order_no IS NOT NULL AND TRIM(sp_order_no) != ''"
            )
            max_sp_returns = cur.fetchone()[0]
            cur.execute("SELECT number FROM invoices WHERE number IS NOT NULL")
            max_mm = None
            max_mm_num = None
            for (num,) in cur.fetchall():
                if not num:
                    continue
                digits = invoice_digits(num)
                if not digits:
                    continue
                try:
                    val = int(digits)
                except ValueError:
                    continue
                if max_mm_num is None or val > max_mm_num:
                    max_mm_num = val
                    max_mm = num
            cur.execute(
                "SELECT MAX(dtposted) FROM bank_transactions WHERE dtposted IS NOT NULL"
            )
            max_bank_date = cur.fetchone()[0]
        finally:
            conn.close()

        bank_msg = "Nema podataka"
        if max_bank_date:
            try:
                dt = pd.to_datetime(max_bank_date, errors="coerce")
                if not pd.isna(dt):
                    bank_msg = dt.strftime("%d-%m-%Y")
                else:
                    bank_msg = str(max_bank_date)
            except Exception:
                bank_msg = str(max_bank_date)

        messagebox.showinfo(
            "Zadnji brojevi",
            "SP narudzbe (max): {0}\n"
            "Minimax racun (max): {1}\n"
            "SP uplate (max): {2}\n"
            "SP preuzimanja (max): {3}\n"
            "Zadnji broj izvoda je za {4}".format(
                max_sp_orders or "-",
                max_mm or "-",
                max_sp_payments or "-",
                max_sp_returns or "-",
                bank_msg,
            ),
        )

    def run_action(action_fn):
        conn = get_conn()
        try:
            action_fn(conn)
            messagebox.showinfo("OK", "Akcija zavrsena.")
        except Exception as exc:
            log_app_error("run_action", str(exc))
            messagebox.showerror("Greska", str(exc))
        finally:
            conn.close()
        refresh_dashboard()

    def set_baseline_lock():
        if state.get("baseline_locked"):
            return
        if not messagebox.askyesno(
            "Potvrda",
            "Ovo ce zakljucati trenutno stanje baze kao pocetno.\n"
            "Nakon toga su moguci samo novi uvozi.\n"
            "Zelite li nastaviti?",
        ):
            return
        conn = get_conn()
        try:
            set_app_state(conn, "baseline_locked", "1")
            set_app_state(conn, "baseline_locked_at", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        finally:
            conn.close()
        load_baseline_lock()
        update_baseline_ui()

    def unlock_baseline_with_password():
        typed = reset_pass_var.get().strip()
        if not typed:
            messagebox.showerror("Greska", "Unesi lozinku za reset.")
            return
        conn = get_conn()
        try:
            stored_hash = get_app_state(conn, "reset_password_hash")
        finally:
            conn.close()
        if not stored_hash:
            messagebox.showerror("Greska", "Lozinka za reset nije postavljena.")
            return
        if hash_password(typed) != stored_hash:
            messagebox.showerror("Greska", "Pogresna lozinka.")
            return
        conn = get_conn()
        try:
            set_app_state(conn, "baseline_locked", "0")
            set_app_state(conn, "baseline_locked_at", "")
        finally:
            conn.close()
        load_baseline_lock()
        update_baseline_ui()
        messagebox.showinfo("OK", "Baza je otkljucana za uvoz dokumenata.")
        update_cartice_ui()

    def set_reset_password():
        value = reset_pass_var.get().strip()
        if not value:
            messagebox.showerror("Greska", "Unesi lozinku za reset.")
            return
        conn = get_conn()
        try:
            set_app_state(conn, "reset_password_hash", hash_password(value))
        finally:
            conn.close()
        reset_pass_var.set("")
        messagebox.showinfo("OK", "Lozinka za reset je sacuvana.")

    def run_full_reset():
        if not messagebox.askyesno(
            "Potvrda",
            "Ovo ce napraviti backup i obrisati kompletnu bazu.\n"
            "Zelite li nastaviti?",
        ):
            return
        conn = get_conn()
        try:
            stored_hash = get_app_state(conn, "reset_password_hash")
        finally:
            conn.close()
        if not stored_hash:
            messagebox.showerror("Greska", "Nije postavljena lozinka za reset.")
            return
        typed = reset_pass_var.get().strip()
        if not typed or hash_password(typed) != stored_hash:
            messagebox.showerror("Greska", "Pogresna lozinka za reset.")
            return
        backup_path = state["db_path"].with_suffix(
            f".bak-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        )
        try:
            if state["db_path"].exists():
                shutil.copy2(state["db_path"], backup_path)
                state["db_path"].unlink()
        except Exception as exc:
            messagebox.showerror("Greska", f"Backup/reset neuspjesan: {exc}")
            return
        conn = get_conn()
        conn.close()
        conn = get_conn()
        conn.close()
        load_baseline_lock()
        update_baseline_ui()
        refresh_dashboard()
        messagebox.showinfo("OK", f"Reset zavrsen. Backup: {backup_path}")

    def run_backup_only():
        typed = reset_pass_var.get().strip()
        if not typed:
            messagebox.showerror("Greska", "Unesi lozinku za reset.")
            return
        conn = get_conn()
        try:
            stored_hash = get_app_state(conn, "reset_password_hash")
        finally:
            conn.close()
        if not stored_hash:
            messagebox.showerror("Greska", "Lozinka za reset nije postavljena.")
            return
        if hash_password(typed) != stored_hash:
            messagebox.showerror("Greska", "Pogresna lozinka.")
            return
        backup_path = state["db_path"].with_suffix(
            f".bak-only-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        )
        try:
            shutil.copy2(state["db_path"], backup_path)
        except Exception as exc:
            messagebox.showerror("Greska", f"Backup neuspjesan: {exc}")
            return
        messagebox.showinfo("OK", f"Backup spreman: {backup_path}")

    def run_export_bank_refunds():
        conn = get_conn()
        try:
            extract_bank_refunds(conn)
            cols, rows = report_bank_refunds_extracted(conn, return_rows=True)
            out_path = write_report(cols, rows, Path("exports"), "bank-refunds-extracted", "xlsx")
            messagebox.showinfo("OK", f"Export zavrsen: {out_path}")
        except Exception as exc:
            messagebox.showerror("Greska", str(exc))
        finally:
            conn.close()
        refresh_dashboard()

    def run_export_refund_category(category: str):
        conn = get_conn()
        try:
            cols, rows = report_refund_items_category(
                conn,
                category,
                state.get("period_days"),
                state.get("period_start"),
                state.get("period_end"),
                return_rows=True,
            )
            out_path = write_report(
                cols,
                rows,
                Path("exports"),
                f"refund-items-{category.replace(' ', '-').lower()}",
                "xlsx",
            )
            try:
                os.startfile(out_path)  # type: ignore[attr-defined]
            except Exception:
                messagebox.showinfo("Info", f"Export zavrsen: {out_path}")
        except Exception as exc:
            messagebox.showerror("Greska", str(exc))
        finally:
            conn.close()
        refresh_dashboard()

    def run_export_refunds_full():
        conn = get_conn()
        try:
            extract_bank_refunds(conn)
            period_days = state.get("period_days")
            start = state.get("period_start")
            end = state.get("period_end")

            items_totals = build_refund_item_totals(conn, period_days, start, end)
            items_rows = sorted(items_totals.items(), key=lambda x: x[1], reverse=True)
            items_df = pd.DataFrame(items_rows, columns=["sku", "qty_refund"])

            by_cat = {}
            for sku, qty in items_rows:
                cat = kategorija_za_sifru(str(sku))
                by_cat[cat] = by_cat.get(cat, 0.0) + float(qty or 0)
            cat_rows = sorted(by_cat.items(), key=lambda x: x[1], reverse=True)
            cat_df = pd.DataFrame(cat_rows, columns=["category", "qty_refund"])

            refunds = conn.execute(
                "SELECT br.bank_txn_id, bt.dtposted, bt.amount, bt.purpose, bt.payee_name, "
                "bt.stmt_number, br.invoice_no, br.invoice_no_digits, br.invoice_no_source, br.reason "
                "FROM bank_refunds br "
                "JOIN bank_transactions bt ON bt.id = br.bank_txn_id"
            ).fetchall()

            inv_by_number = {}
            inv_by_digits = {}
            for inv_id, inv_no, inv_name in conn.execute(
                "SELECT id, number, customer_name FROM invoices"
            ).fetchall():
                inv_by_number[str(inv_no or "")] = (int(inv_id), inv_name)
                digits = invoice_digits(inv_no)
                if digits:
                    inv_by_digits.setdefault(digits, set()).add((int(inv_id), inv_name))

            storno_map = {
                int(r[0]): int(r[1])
                for r in conn.execute(
                    "SELECT storno_invoice_id, original_invoice_id FROM invoice_storno"
                ).fetchall()
            }
            order_by_invoice = {
                int(inv_id): int(order_id)
                for inv_id, order_id in conn.execute(
                    "SELECT invoice_id, order_id FROM invoice_matches"
                ).fetchall()
            }
            order_name = {
                int(oid): name
                for oid, name in conn.execute(
                    "SELECT id, customer_name FROM orders"
                ).fetchall()
            }
            items_by_order = {}
            for oid, sku, qty in conn.execute(
                "SELECT order_id, product_code, qty FROM order_items"
            ).fetchall():
                if not sku:
                    continue
                items_by_order.setdefault(int(oid), []).append((str(sku), float(qty or 0)))

            detail_rows = []
            for (
                bank_txn_id,
                dtposted,
                amount,
                purpose,
                payee_name,
                stmt_number,
                inv_no,
                inv_digits,
                inv_source,
                reason,
            ) in refunds:
                inv_id = None
                inv_name = None
                if inv_no and str(inv_no) in inv_by_number:
                    inv_id, inv_name = inv_by_number[str(inv_no)]
                elif inv_digits and inv_digits in inv_by_digits and len(inv_by_digits[inv_digits]) == 1:
                    inv_id, inv_name = next(iter(inv_by_digits[inv_digits]))
                if inv_id in storno_map:
                    inv_id = storno_map[inv_id]
                order_id = order_by_invoice.get(inv_id) if inv_id else None
                customer = None
                if order_id:
                    customer = order_name.get(order_id)
                if not customer:
                    customer = inv_name or payee_name
                sku_list = ""
                if order_id and order_id in items_by_order:
                    sku_list = "; ".join(
                        f"{sku} x{int(qty)}" for sku, qty in items_by_order[order_id]
                    )
                detail_rows.append(
                    (
                        bank_txn_id,
                        dtposted,
                        amount,
                        stmt_number,
                        customer,
                        purpose,
                        inv_no,
                        inv_digits,
                        inv_source,
                        reason,
                        inv_id,
                        order_id,
                        sku_list,
                    )
                )
            detail_df = pd.DataFrame(
                detail_rows,
                columns=[
                    "bank_txn_id",
                    "dtposted",
                    "amount",
                    "stmt_number",
                    "customer_name",
                    "purpose",
                    "invoice_no",
                    "invoice_no_digits",
                    "invoice_no_source",
                    "reason",
                    "invoice_id",
                    "order_id",
                    "sku_list",
                ],
            )

            out_path = Path("exports") / "refunds-full.xlsx"
            with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
                items_df.to_excel(writer, sheet_name="items", index=False)
                cat_df.to_excel(writer, sheet_name="categories", index=False)
                detail_df.to_excel(writer, sheet_name="refunds", index=False)
            try:
                os.startfile(out_path)  # type: ignore[attr-defined]
            except Exception:
                messagebox.showinfo("Info", f"Export zavrsen: {out_path}")
        except Exception as exc:
            messagebox.showerror("Greska", str(exc))
        finally:
            conn.close()
        refresh_dashboard()

    def run_export_unpicked_full():
        conn = get_conn()
        try:
            period_days = state.get("unpicked_period_days")
            start = state.get("unpicked_period_start")
            end = state.get("unpicked_period_end")
            stats = get_unpicked_stats(conn, period_days, start, end)
            items = get_unpicked_top_items(conn, None, period_days, start, end)
            categories = get_unpicked_category_totals(conn, period_days, start, end)
            orders = get_unpicked_orders_list(conn, period_days, start, end)

            items_rows = []
            for sku, qty, net in items:
                items_rows.append((sku, qty, net, kategorija_za_sifru(str(sku))))
            items_df = pd.DataFrame(
                items_rows,
                columns=["sku", "qty", "net_total", "category"],
            )
            cat_df = pd.DataFrame(
                categories,
                columns=["category", "qty", "net_total"],
            )
            tracking_codes = [row[1] for row in orders if row[1]]
            summary_map = {}
            if tracking_codes:
                placeholders = ",".join("?" * len(tracking_codes))
                for row in conn.execute(
                    "SELECT tracking_code, received_at, first_out_for_delivery_at, "
                    "delivery_attempts, failure_reasons, returned_at, days_to_first_attempt, "
                    "has_attempt_before_return, has_returned, anomalies, last_status, last_status_at "
                    f"FROM tracking_summary WHERE tracking_code IN ({placeholders})",
                    tracking_codes,
                ).fetchall():
                    summary_map[row[0]] = row[1:]

            analysis_metrics = []
            reasons_count = {}
            anomalies_count = {}
            attempts_count = {"0": 0, "1": 0, "2": 0, "3+": 0}
            days_to_first = []
            out_to_return = []
            no_attempt_before_return = 0
            not_returned = 0
            one_attempt = 0
            late_first_attempt = 0

            for code in tracking_codes:
                summary = summary_map.get(code)
                if not summary:
                    continue
                received_at = summary[0]
                first_out = summary[1]
                attempts = summary[2] or 0
                reasons = summary[3] or ""
                returned_at = summary[4]
                days_first = summary[5]
                has_attempt_before_return = summary[6]
                has_returned = summary[7]
                anomalies = summary[8] or ""

                if has_returned == 0:
                    not_returned += 1
                if has_returned == 1 and has_attempt_before_return == 0:
                    no_attempt_before_return += 1
                if attempts == 1:
                    one_attempt += 1
                if attempts >= 3:
                    attempts_count["3+"] += 1
                elif attempts == 2:
                    attempts_count["2"] += 1
                elif attempts == 1:
                    attempts_count["1"] += 1
                else:
                    attempts_count["0"] += 1

                if days_first is not None:
                    try:
                        days_first_val = float(days_first)
                        days_to_first.append(days_first_val)
                        if days_first_val > 2:
                            late_first_attempt += 1
                    except (TypeError, ValueError):
                        pass

                if first_out and returned_at:
                    try:
                        dt_out = pd.to_datetime(first_out)
                        dt_ret = pd.to_datetime(returned_at)
                        out_to_return.append((dt_ret - dt_out).total_seconds() / 86400.0)
                    except Exception:
                        pass

                for reason in [r.strip() for r in reasons.split(";") if r.strip()]:
                    reasons_count[reason] = reasons_count.get(reason, 0) + 1
                for anomaly in [a.strip() for a in anomalies.split(";") if a.strip()]:
                    anomalies_count[anomaly] = anomalies_count.get(anomaly, 0) + 1

            avg_days_first = sum(days_to_first) / len(days_to_first) if days_to_first else None
            median_days_first = statistics.median(days_to_first) if days_to_first else None
            avg_out_to_return = sum(out_to_return) / len(out_to_return) if out_to_return else None
            median_out_to_return = statistics.median(out_to_return) if out_to_return else None

            top_reason = None
            if reasons_count:
                top_reason = max(reasons_count.items(), key=lambda x: x[1])[0]

            analysis_metrics.extend(
                [
                    ("posiljke_ukupno", len(tracking_codes)),
                    ("posiljke_nisu_vracene", not_returned),
                    ("posiljke_bez_pokusaja_prije_vracanja", no_attempt_before_return),
                    ("posiljke_samo_jedan_pokusaj", one_attempt),
                    ("posiljke_prvi_pokusaj_>2_dana", late_first_attempt),
                    ("prosjek_dana_do_prvog_pokusaja", avg_days_first),
                    ("medijan_dana_do_prvog_pokusaja", median_days_first),
                    ("prosjek_dana_od_zaduzenja_do_vracanja", avg_out_to_return),
                    ("medijan_dana_od_zaduzenja_do_vracanja", median_out_to_return),
                    ("najcesci_razlog_nedostavljanja", top_reason or ""),
                ]
            )

            orders_rows = []
            for row in orders:
                (
                    sp_order_no,
                    tracking_code,
                    customer_name,
                    phone,
                    email,
                    city,
                    status,
                    created_at,
                    picked_up_at,
                    delivered_at,
                    net_total,
                ) = row
                summary = summary_map.get(tracking_code, (None,) * 11)
                tracking_url = tracking_public_url(tracking_code) if tracking_code else None
                orders_rows.append(
                    (
                        sp_order_no,
                        tracking_code,
                        tracking_url,
                        customer_name,
                        phone,
                        email,
                        city,
                        status,
                        created_at,
                        picked_up_at,
                        delivered_at,
                        net_total,
                    )
                    + summary
                )

            orders_df = pd.DataFrame(
                orders_rows,
                columns=[
                    "sp_order_no",
                    "tracking_code",
                    "tracking_url",
                    "customer_name",
                    "phone",
                    "email",
                    "city",
                    "status",
                    "created_at",
                    "picked_up_at",
                    "delivered_at",
                    "net_total",
                    "received_at",
                    "first_out_for_delivery_at",
                    "delivery_attempts",
                    "failure_reasons",
                    "returned_at",
                    "days_to_first_attempt",
                    "has_attempt_before_return",
                    "has_returned",
                    "anomalies",
                    "last_status",
                    "last_status_at",
                ],
            )
            summary_df = pd.DataFrame(
                [
                    ("unpicked_orders", stats["unpicked_orders"]),
                    ("lost_sales", stats["lost_sales"]),
                    ("repeat_customers", stats["repeat_customers"]),
                ],
                columns=["metric", "value"],
            )
            analysis_df = pd.DataFrame(analysis_metrics, columns=["metric", "value"])
            reasons_df = pd.DataFrame(
                sorted(reasons_count.items(), key=lambda x: x[1], reverse=True),
                columns=["reason", "count"],
            )
            anomalies_df = pd.DataFrame(
                sorted(anomalies_count.items(), key=lambda x: x[1], reverse=True),
                columns=["anomaly", "count"],
            )
            attempts_df = pd.DataFrame(
                [(k, v) for k, v in attempts_count.items()],
                columns=["attempts", "count"],
            )

            out_path = Path("exports") / "unpicked-full.xlsx"
            with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
                summary_df.to_excel(writer, sheet_name="summary", index=False)
                items_df.to_excel(writer, sheet_name="items", index=False)
                cat_df.to_excel(writer, sheet_name="categories", index=False)
                orders_df.to_excel(writer, sheet_name="orders", index=False)
                analysis_df.to_excel(writer, sheet_name="tracking-analysis", index=False)
                reasons_df.to_excel(writer, sheet_name="tracking-reasons", index=False)
                anomalies_df.to_excel(writer, sheet_name="tracking-anomalies", index=False)
                attempts_df.to_excel(writer, sheet_name="tracking-attempts", index=False)
            try:
                os.startfile(out_path)  # type: ignore[attr-defined]
            except Exception:
                messagebox.showinfo("Info", f"Export zavrsen: {out_path}")
        except Exception as exc:
            messagebox.showerror("Greska", str(exc))
        finally:
            conn.close()

    def run_export_audit():
        conn = get_conn()
        try:
            imports_df = pd.read_sql_query("SELECT * FROM import_runs ORDER BY imported_at", conn)
            tasks_df = pd.read_sql_query("SELECT * FROM task_progress ORDER BY updated_at DESC", conn)
            actions_df = pd.read_sql_query("SELECT * FROM action_log ORDER BY created_at DESC", conn)
            tracking_df = pd.read_sql_query("SELECT * FROM tracking_summary ORDER BY last_fetched_at DESC", conn)

            log_path = Path("exports") / "tracking-log.csv"
            if log_path.exists():
                tracking_log_df = pd.read_csv(log_path)
            else:
                tracking_log_df = pd.DataFrame(
                    columns=["timestamp", "tracking_code", "url", "status_code", "latency_ms", "result", "error"]
                )

            err_path = Path("exports") / "app-errors.log"
            if err_path.exists():
                error_lines = err_path.read_text(encoding="utf-8").splitlines()
                errors_df = pd.DataFrame(error_lines, columns=["error"])
            else:
                errors_df = pd.DataFrame(columns=["error"])

            out_path = Path("exports") / "audit.xlsx"
            with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
                imports_df.to_excel(writer, sheet_name="imports", index=False)
                tasks_df.to_excel(writer, sheet_name="tasks", index=False)
                actions_df.to_excel(writer, sheet_name="actions", index=False)
                tracking_df.to_excel(writer, sheet_name="tracking_summary", index=False)
                tracking_log_df.to_excel(writer, sheet_name="tracking_log", index=False)
                errors_df.to_excel(writer, sheet_name="errors", index=False)
            try:
                os.startfile(out_path)  # type: ignore[attr-defined]
            except Exception:
                messagebox.showinfo("Info", f"Audit export: {out_path}")
        except Exception as exc:
            log_app_error("audit_export", str(exc))
            messagebox.showerror("Greska", str(exc))
        finally:
            conn.close()

    def poll_tracking_progress():
        if not state.get("tracking_polling"):
            return
        info = get_progress_info("tracking")
        if not info:
            tracking_status_var.set("Dexpress: cekanje...")
            app.after(1000, poll_tracking_progress)
            return
        total = info.get("total", 0)
        processed = info.get("processed", 0)
        if total <= 0:
            tracking_status_var.set("Dexpress: nema podataka")
            state["tracking_polling"] = False
            return
        pct = int((processed / total) * 100)
        tracking_status_var.set(f"Dexpress: {processed}/{total} ({pct}%)")
        if processed >= total:
            tracking_status_var.set(f"Dexpress: zavrseno ({processed}/{total})")
            state["tracking_polling"] = False
            refresh_dashboard()
            return
        app.after(1000, poll_tracking_progress)

    def run_dexpress_tracking():
        state["tracking_polling"] = True
        tracking_status_var.set("Dexpress: pokrenuto...")
        try:
            batch = int(tracking_batch_var.get().strip() or "20")
            if batch <= 0:
                batch = 20
        except ValueError:
            batch = 20
        force_flag = 1 if tracking_force_var.get() else 0
        run_action_async_process(
            run_tracking_process,
            [str(state["db_path"]), batch, force_flag],
            "Dexpress analiza",
            progress_task="tracking",
        )
        app.after(1000, poll_tracking_progress)

    test_session_rows = []
    test_session_index = 0
    test_session_log = []
    test_session_path = None

    def _update_test_view():
        nonlocal test_session_index
        txt_test_order.delete("1.0", "end")
        txt_test_invoice.delete("1.0", "end")
        if not test_session_rows:
            lbl_test_status.configure(text="Nema podataka za test.")
            return
        if test_session_index >= len(test_session_rows):
            lbl_test_status.configure(text="Sesija zavrsena.")
            return
        item = test_session_rows[test_session_index]
        lbl_test_status.configure(
            text=f"{test_session_index + 1}/{len(test_session_rows)} (score {item['score']})"
        )
        txt_test_order.insert(
            "end",
            f"SP broj: {item['sp_order_no']}\n"
            f"Ime: {item['order_name']}\n"
            f"Datum: {item['order_date']}\n"
            f"Iznos (RSD): {format_amount_rsd(item['order_amount'])}\n",
        )
        txt_test_invoice.insert(
            "end",
            f"Racun: {item['invoice_no']}\n"
            f"Kupac: {item['invoice_name']}\n"
            f"Datum: {item['invoice_date']}\n"
            f"Iznos (RSD): {format_amount_rsd(item['invoice_amount'])}\n",
        )

    def _save_test_log():
        if not test_session_path:
            return
        df = pd.DataFrame(test_session_log)
        df.to_excel(test_session_path, index=False)

    def _record_decision(decision: str):
        nonlocal test_session_index
        if not test_session_rows or test_session_index >= len(test_session_rows):
            return
        item = test_session_rows[test_session_index]
        test_session_log.append(
            {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "match_id": item["match_id"],
                "score": item["score"],
                "sp_order_no": item["sp_order_no"],
                "order_name": item["order_name"],
                "order_date": item["order_date"],
                "order_amount": item["order_amount"],
                "invoice_no": item["invoice_no"],
                "invoice_name": item["invoice_name"],
                "invoice_date": item["invoice_date"],
                "invoice_amount": item["invoice_amount"],
                "decision": decision,
            }
        )
        _save_test_log()
        test_session_index += 1
        _update_test_view()

    def start_test_session(mode: str):
        nonlocal test_session_rows, test_session_index, test_session_log, test_session_path
        try:
            size = int(test_size_var.get().strip() or "30")
        except ValueError:
            size = 30
        if size <= 0:
            size = 30
        conn = get_conn()
        try:
            if mode == "all":
                test_session_rows = load_all_match_samples(conn, size)
            else:
                test_session_rows = load_review_samples(conn, size)
        finally:
            conn.close()
        if not test_session_rows:
            msg = "Nema sumnjivih matchova za test." if mode != "all" else "Nema matchova za test."
            messagebox.showinfo("Info", msg)
            lbl_test_status.configure(text="Nema podataka.")
            return
        test_session_index = 0
        test_session_log = []
        exports_dir = Path("exports")
        exports_dir.mkdir(parents=True, exist_ok=True)
        test_session_path = exports_dir / f"match-test-log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        _update_test_view()

    def stop_test_session():
        nonlocal test_session_rows, test_session_index
        test_session_rows = []
        test_session_index = 0
        lbl_test_status.configure(text="Sesija prekinuta.")
        txt_test_order.delete("1.0", "end")
        txt_test_invoice.delete("1.0", "end")

    def run_reset_minimax_matches():
        if not messagebox.askyesno(
            "Potvrda",
            "Ovo ce obrisati sva Minimax uparivanja i kandidate.\n"
            "Zelite li nastaviti?",
        ):
            return
        run_action(reset_minimax_matches)

    def run_action_async(action_fn, label: str):
        for btn in action_buttons:
            btn.configure(state="disabled")
        status_var.set(f"Radi: {label}...")
        progress.configure(mode="indeterminate")
        progress_pct_var.set("")
        progress.start()

        result = {"error": None}

        def worker():
            conn = get_conn()
            try:
                action_fn(conn)
            except Exception as exc:
                result["error"] = exc
            finally:
                conn.close()
            app.after(0, on_done)

        def on_done():
            progress.stop()
            for btn in action_buttons:
                btn.configure(state="normal")
            if result["error"]:
                log_app_error(label, str(result["error"]))
                messagebox.showerror("Greska", str(result["error"]))
                status_var.set("Greska.")
            else:
                status_var.set("Zavrseno.")
                refresh_dashboard()
                refresh_poslovanje_lists()

        threading.Thread(target=worker, daemon=True).start()

    def run_action_async_process(fn, args, label: str, progress_task: str | None = None):
        for btn in action_buttons:
            btn.configure(state="disabled")
        status_var.set(f"Radi: {label}...")
        if progress_task:
            progress.configure(mode="determinate")
            progress.set(0)
            progress_pct_var.set("Napredak: 0%")
            progress_eta_var.set("ETA: --")
            state["progress_start"] = time.time()
        else:
            progress.configure(mode="indeterminate")
            progress_pct_var.set("")
            progress_eta_var.set("")
            progress.start()

        future = executor.submit(fn, *args)

        def poll():
            if progress_task:
                info = get_progress_info(progress_task)
                if info is not None:
                    total = info["total"]
                    processed = info["processed"]
                    pct = min(1.0, processed / total) if total > 0 else 0.0
                    if processed > 0 and pct < 0.01:
                        pct_display = 0.01
                    else:
                        pct_display = pct
                    progress.set(pct_display)
                    if processed == 0:
                        progress_pct_var.set("Priprema...")
                        progress_eta_var.set("ETA: --")
                    else:
                        progress_pct_var.set(f"Napredak: {int(pct * 100)}%")
                        elapsed = time.time() - state.get("progress_start", time.time())
                        eta = elapsed * (total - processed) / processed if processed > 0 else None
                        progress_eta_var.set(f"ETA: {format_eta(eta)}")
            if not future.done():
                app.after(5000, poll)
                return
            if progress_task:
                progress.set(1)
                progress_pct_var.set("Napredak: 100%")
                progress_eta_var.set("ETA: 0s")
            else:
                progress.stop()
            for btn in action_buttons:
                btn.configure(state="normal")
            try:
                future.result()
            except Exception as exc:
                log_app_error(label, str(exc))
                messagebox.showerror("Greska", str(exc))
                status_var.set("Greska.")
                return
            status_var.set("Zavrseno.")
            refresh_dashboard()
            refresh_poslovanje_lists()

        poll()

    top = ctk.CTkFrame(app)
    top.pack(fill="x", padx=12, pady=8)

    ctk.CTkButton(top, text="Osvjezi", command=refresh_dashboard).pack(side="left", padx=6)
    task_status_var = ctk.StringVar(value="Task: idle")
    last_error_var = ctk.StringVar(value="Zadnja greska: -")
    ctk.CTkLabel(top, textvariable=task_status_var).pack(side="left", padx=12)
    ctk.CTkLabel(top, textvariable=last_error_var).pack(side="left", padx=12)

    def open_error_log():
        path = Path("exports") / "app-errors.log"
        if not path.exists():
            messagebox.showinfo("Info", "Nema loga gresaka.")
            return
        try:
            os.startfile(path)  # type: ignore[attr-defined]
        except Exception:
            messagebox.showinfo("Info", f"Log gresaka: {path}")

    ctk.CTkButton(top, text="Otvori log gresaka", command=open_error_log).pack(
        side="left", padx=6
    )

    tabs = ctk.CTkTabview(app)
    tabs.pack(fill="both", expand=True, padx=12, pady=8)

    tab_dashboard = tabs.add("Dashboard")
    tab_prodaja = tabs.add("Prodaja")
    tabs.add("Marze")
    tab_troskovi = tabs.add("Troskovi")
    tab_povrati = tabs.add("Povrati")
    tab_nepreuzete = tabs.add("Nepreuzete")
    tab_test = tabs.add("Test uparivanja")
    tabs.add("Kupci")
    tab_poslovanje = tabs.add("Poslovanje")
    tab_settings = tabs.add("Podesavanja aplikacije")

    prodaja_ops = ctk.CTkFrame(tab_prodaja)
    prodaja_ops.pack(fill="x", padx=10, pady=10)

    ctk.CTkLabel(prodaja_ops, text="Period:").pack(side="left", padx=(6, 4))
    prodaja_period_var = ctk.StringVar(value="12 mjeseci")
    period_mapping = {
        "Svo vrijeme": None,
        "3 mjeseca": 90,
        "6 mjeseci": 180,
        "12 mjeseci": 360,
        "24 mjeseca": 720,
    }

    def on_prodaja_period_change(choice: str):
        state["prodaja_period_days"] = period_mapping.get(choice)
        state["prodaja_period_start"] = None
        state["prodaja_period_end"] = None
        prodaja_custom_from_var.set("")
        prodaja_custom_to_var.set("")
        prodaja_period_summary_var.set("")
        refresh_prodaja_tab()

    prodaja_period_menu = ctk.CTkOptionMenu(
        prodaja_ops,
        values=list(period_mapping.keys()),
        variable=prodaja_period_var,
        command=on_prodaja_period_change,
    )
    prodaja_period_menu.pack(side="left", padx=4)

    prodaja_custom_from_var = ctk.StringVar(value="")
    prodaja_custom_to_var = ctk.StringVar(value="")
    ctk.CTkLabel(prodaja_ops, text="Od (YYYY-MM-DD):").pack(side="left", padx=(12, 4))
    prodaja_from_entry = ctk.CTkEntry(prodaja_ops, width=120, textvariable=prodaja_custom_from_var)
    prodaja_from_entry.pack(side="left", padx=4)
    ctk.CTkLabel(prodaja_ops, text="Do (YYYY-MM-DD):").pack(side="left", padx=(12, 4))
    prodaja_to_entry = ctk.CTkEntry(prodaja_ops, width=120, textvariable=prodaja_custom_to_var)
    prodaja_to_entry.pack(side="left", padx=4)

    def apply_prodaja_custom():
        start = _parse_user_date(prodaja_custom_from_var.get())
        end = _parse_user_date(prodaja_custom_to_var.get())
        if start is None or end is None:
            messagebox.showerror("Greska", "Unesi ispravan period (YYYY-MM-DD).")
            return
        if end < start:
            messagebox.showerror("Greska", "Datum završetka mora biti poslije početka.")
            return
        state["prodaja_period_days"] = None
        state["prodaja_period_start"] = start
        state["prodaja_period_end"] = end
        prodaja_period_summary_var.set("")
        refresh_prodaja_tab()

    ctk.CTkButton(prodaja_ops, text="Primijeni period", command=apply_prodaja_custom).pack(
        side="left", padx=(12, 4)
    )
    prodaja_period_summary_var = ctk.StringVar(value="")
    ctk.CTkLabel(prodaja_ops, textvariable=prodaja_period_summary_var).pack(
        side="left", padx=12
    )

    prodaja_content = ctk.CTkFrame(tab_prodaja)
    prodaja_content.pack(fill="both", expand=True, padx=10, pady=10)
    prodaja_left = ctk.CTkFrame(prodaja_content)
    prodaja_left.pack(side="left", fill="both", expand=True, padx=6, pady=6)
    prodaja_right = ctk.CTkFrame(prodaja_content)
    prodaja_right.pack(side="left", fill="both", expand=True, padx=6, pady=6)

    ctk.CTkLabel(prodaja_left, text="Top 5 potencijalni gubici (neto):").pack(
        anchor="w", padx=6, pady=(6, 4)
    )
    prodaja_top5_txt = ctk.CTkTextbox(prodaja_left, height=260)
    prodaja_top5_txt.pack(fill="both", expand=True, padx=6, pady=(0, 6))
    prodaja_top5_txt.configure(state="disabled")

    ctk.CTkLabel(prodaja_right, text="Top 10 sve vrijeme (neto gubitak):").pack(
        anchor="w", padx=6, pady=(6, 4)
    )
    prodaja_top10_txt = ctk.CTkTextbox(prodaja_right, height=260)
    prodaja_top10_txt.pack(fill="both", expand=True, padx=6, pady=(0, 6))
    prodaja_top10_txt.configure(state="disabled")

    prodaja_hints = ctk.CTkLabel(
        prodaja_right,
        text="Korišteni period filtrira procjene gubitka po dostupnosti; top 10 sve vrijeme ignorira period.",
        wraplength=320,
        justify="left",
    )
    prodaja_hints.pack(anchor="w", padx=6, pady=(0, 6))

    def _format_prodaja_entry(idx: int, row: dict) -> str:
        loss = format_amount(row.get("Procjena gubitka neto", 0))
        days = int(row.get("Dani bez zalihe", 0) or 0)
        avg_net = format_amount(row.get("Prosek neto", 0))
        art = row.get("Artikal", "")
        sku = row.get("SKU", "")
        return f"{idx}. {sku} — {art}\n   Gubitak: {loss}, Dani: {days}, Prosek neto: {avg_net}"

    def _render_prodaja_textbox(widget: ctk.CTkTextbox, lines: list[str]):
        widget.configure(state="normal")
        widget.delete("0.0", "end")
        widget.insert("0.0", "\n\n".join(lines))
        widget.configure(state="disabled")

    def refresh_prodaja_tab():
        df = _load_prodaja_dataframe()
        start, end = _resolve_prodaja_period()
        if start and end:
            prodaja_period_summary_var.set(f"Period: {start.isoformat()} → {end.isoformat()}")
        elif start:
            prodaja_period_summary_var.set(f"Period: od {start.isoformat()}")
        elif end:
            prodaja_period_summary_var.set(f"Period: do {end.isoformat()}")
        else:
            prodaja_period_summary_var.set("Period: svi podaci")

        filtered = _filter_prodaja_by_period(df, start, end)
        if filtered.empty:
            _render_prodaja_textbox(
                prodaja_top5_txt, ["Nema podataka za odabrani period."]
            )
        else:
            top5 = (
                filtered.sort_values("Procjena gubitka neto", ascending=False)
                .head(5)
                .to_dict("records")
            )
            lines = [_format_prodaja_entry(i + 1, row) for i, row in enumerate(top5)]
            _render_prodaja_textbox(prodaja_top5_txt, lines)

        if df.empty:
            _render_prodaja_textbox(
                prodaja_top10_txt, ["Nema dostupnih podataka za top 10."]
            )
        else:
            top10 = (
                df.sort_values("Procjena gubitka neto", ascending=False)
                .head(10)
                .to_dict("records")
            )
            lines10 = [_format_prodaja_entry(i + 1, row) for i, row in enumerate(top10)]
            _render_prodaja_textbox(prodaja_top10_txt, lines10)

    refresh_prodaja_tab()

    settings_body = ctk.CTkFrame(tab_settings)
    settings_body.pack(fill="both", expand=True, padx=12, pady=12)
    ctk.CTkLabel(settings_body, text="Baza podataka").pack(anchor="w", padx=6, pady=(6, 4))
    db_row = ctk.CTkFrame(settings_body)
    db_row.pack(fill="x", padx=6, pady=(0, 10))
    ctk.CTkLabel(db_row, text="DB:").pack(side="left", padx=(6, 4))
    ent_db = ctk.CTkEntry(db_row, width=600)
    ent_db.insert(0, str(db_path))
    ent_db.configure(state="readonly")
    ent_db.pack(side="left", padx=4)
    ctk.CTkButton(db_row, text="Promijeni", command=set_db_path).pack(side="left", padx=6)

    ctk.CTkLabel(settings_body, text="Uvoz (pocetno stanje)").pack(anchor="w", padx=6, pady=(6, 4))
    ctk.CTkLabel(
        settings_body,
        text="Workflow: 1) SP Narudzbe 2) Minimax 3) SP Uplate 4) Banka XML 5) SP Preuzimanja",
    ).pack(anchor="w", padx=6, pady=(0, 6))
    base_imports = ctk.CTkFrame(settings_body)
    base_imports.pack(anchor="w", fill="x", padx=6, pady=(0, 10))
    ctk.CTkButton(
        base_imports,
        text="SP Narudzbe",
        command=lambda: run_import_folder(import_sp_orders, "SP Narudzbe (folder)", "*.xlsx"),
    ).pack(anchor="w", pady=2)
    ctk.CTkButton(
        base_imports,
        text="Minimax",
        command=lambda: run_import_folder(import_minimax, "Minimax (folder)", "*.xlsx"),
    ).pack(anchor="w", pady=2)
    ctk.CTkButton(
        base_imports,
        text="SP Uplate",
        command=lambda: run_import_folder(import_sp_payments, "SP Uplate (folder)", "*.xlsx"),
    ).pack(anchor="w", pady=2)
    ctk.CTkButton(
        base_imports,
        text="Banka XML",
        command=lambda: run_import_folder(import_bank_xml, "Banka XML (folder)", "*.xml"),
    ).pack(anchor="w", pady=2)
    ctk.CTkButton(
        base_imports,
        text="SP Preuzimanja",
        command=lambda: run_import_folder(import_sp_returns, "SP Preuzimanja (folder)", "*.xlsx"),
    ).pack(anchor="w", pady=2)

    ctk.CTkLabel(settings_body, text="Kurs i valuta").pack(anchor="w", padx=6, pady=(6, 4))
    rate_row = ctk.CTkFrame(settings_body)
    rate_row.pack(fill="x", padx=6, pady=(0, 10))
    ctk.CTkLabel(rate_row, text="Kurs RSD->BAM:").pack(side="left", padx=(6, 4))
    rate_var = ctk.StringVar(value="")
    ent_rate = ctk.CTkEntry(rate_row, width=120, textvariable=rate_var)
    ent_rate.pack(side="left", padx=4)

    def sync_rate_entry():
        rate = state.get("rate_rsd_to_bam")
        if rate is None:
            rate_var.set("")
            return
        rate_var.set(f"{rate:.6f}")

    def apply_rate_manual():
        text = rate_var.get().strip().replace(",", ".")
        if not text:
            messagebox.showerror("Greska", "Unesi kurs RSD->BAM.")
            return
        try:
            rate = float(text)
        except ValueError:
            messagebox.showerror("Greska", "Neispravan kurs.")
            return
        conn = get_conn()
        try:
            set_app_state(conn, "rate_rsd_to_bam", f"{rate}")
        finally:
            conn.close()
        state["rate_rsd_to_bam"] = rate
        sync_rate_entry()
        refresh_dashboard()

    def refresh_rate_auto():
        rate, error = refresh_exchange_rate()
        if rate is None:
            messagebox.showwarning(
                "Info",
                "Neuspjelo povlacenje kursa sa CBBH. "
                f"{error or 'Provjeri internet ili unesi kurs rucno.'}",
            )
        sync_rate_entry()
        refresh_dashboard()

    ctk.CTkButton(rate_row, text="Refresh kurs", command=refresh_rate_auto).pack(side="left", padx=4)
    ctk.CTkButton(rate_row, text="Snimi kurs", command=apply_rate_manual).pack(side="left", padx=4)

    currency_row = ctk.CTkFrame(settings_body)
    currency_row.pack(fill="x", padx=6, pady=(0, 10))
    ctk.CTkLabel(currency_row, text="Prikaz valute:").pack(side="left", padx=(6, 4))
    load_currency_mode()
    currency_var = ctk.StringVar(value=state.get("currency_mode", "RSD"))

    def on_currency_change(choice: str):
        state["currency_mode"] = choice
        conn = get_conn()
        try:
            set_app_state(conn, "currency_mode", choice)
        finally:
            conn.close()
        refresh_dashboard()

    currency_menu = ctk.CTkOptionMenu(
        currency_row,
        values=["RSD", "BAM"],
        variable=currency_var,
        command=on_currency_change,
    )
    currency_menu.pack(side="left", padx=4)

    ctk.CTkLabel(settings_body, text="Sigurnost").pack(anchor="w", padx=6, pady=(6, 4))
    security_row = ctk.CTkFrame(settings_body)
    security_row.pack(fill="x", padx=6, pady=(0, 10))
    baseline_status_var = ctk.StringVar(value="")
    ctk.CTkLabel(security_row, text="Status baze:").pack(side="left", padx=(6, 4))
    lbl_baseline = ctk.CTkLabel(security_row, textvariable=baseline_status_var)
    lbl_baseline.pack(side="left", padx=4)
    btn_lock_baseline = ctk.CTkButton(security_row, text="Zakljucaj pocetno stanje", command=set_baseline_lock)
    btn_lock_baseline.pack(side="left", padx=6)
    btn_unlock_baseline = ctk.CTkButton(
        security_row,
        text="Otkljucaj lozinkom",
        command=unlock_baseline_with_password,
    )
    btn_unlock_baseline.pack(side="left", padx=6)
    ctk.CTkButton(
        security_row,
        text="Backup baze",
        command=run_backup_only,
    ).pack(side="left", padx=6)

    reset_row = ctk.CTkFrame(settings_body)
    reset_row.pack(fill="x", padx=6, pady=(0, 10))
    ctk.CTkLabel(reset_row, text="Lozinka za reset:").pack(side="left", padx=(6, 4))
    reset_pass_var = ctk.StringVar(value="")
    ent_reset = ctk.CTkEntry(reset_row, width=160, textvariable=reset_pass_var, show="*")
    ent_reset.pack(side="left", padx=4)
    ctk.CTkButton(reset_row, text="Snimi lozinku", command=set_reset_password).pack(side="left", padx=4)
    ctk.CTkButton(reset_row, text="Potpuni reset aplikacije", command=run_full_reset).pack(side="left", padx=6)

    ctk.CTkLabel(settings_body, text="Audit").pack(anchor="w", padx=6, pady=(6, 4))
    audit_row = ctk.CTkFrame(settings_body)
    audit_row.pack(fill="x", padx=6, pady=(0, 10))
    ctk.CTkButton(audit_row, text="Export audit.xlsx", command=run_export_audit).pack(side="left", padx=6)

    ctk.CTkLabel(settings_body, text="Kartice").pack(anchor="w", padx=6, pady=(6, 4))
    cartice_row = ctk.CTkFrame(settings_body)
    cartice_row.pack(fill="x", padx=6, pady=(0, 10))
    cartice_status_var = ctk.StringVar(value="Kartice: nema uvoza")
    ctk.CTkLabel(cartice_row, textvariable=cartice_status_var).pack(side="left", padx=(6, 4))
    ctk.CTkButton(cartice_row, text="Uvezi kartice", command=import_cartice_data).pack(side="left", padx=4)
    ctk.CTkButton(cartice_row, text="Provjeri zadnje", command=show_latest_kalkulacija).pack(side="left", padx=4)

    def update_baseline_ui():
        locked = state.get("baseline_locked", False)
        locked_at = state.get("baseline_locked_at")
        if locked:
            text = "Zakljucano"
            if locked_at:
                text += f" ({locked_at})"
            baseline_status_var.set(text)
            btn_lock_baseline.configure(state="disabled")
            btn_unlock_baseline.configure(state="normal")
        else:
            baseline_status_var.set("Otkljucano")
            btn_lock_baseline.configure(state="normal")
            btn_unlock_baseline.configure(state="disabled")
        if btn_reset_matches is not None:
            btn_reset_matches.configure(state="disabled" if locked else "normal")

    kpi_frame = ctk.CTkFrame(tab_dashboard)
    kpi_frame.pack(fill="x", padx=10, pady=10)

    def make_kpi(parent, title):
        frame = ctk.CTkFrame(parent, width=200)
        frame.pack(side="left", padx=8, pady=6, fill="x", expand=True)
        ctk.CTkLabel(frame, text=title).pack(pady=(6, 2))
        lbl = ctk.CTkLabel(frame, text="0", font=ctk.CTkFont(size=18, weight="bold"))
        lbl.pack(pady=(0, 6))
        return lbl

    lbl_total_orders = make_kpi(kpi_frame, "Narudzbe")
    lbl_total_revenue = make_kpi(kpi_frame, "Prihod (neto)")
    lbl_returns = make_kpi(kpi_frame, "Nepreuzete")
    lbl_unmatched = make_kpi(kpi_frame, "Neuparene")

    bank_period_frame = ctk.CTkFrame(tab_dashboard)
    bank_period_frame.pack(fill="x", padx=10, pady=(0, 6))
    ctk.CTkLabel(bank_period_frame, text="Prikazi period:").pack(side="left", padx=(6, 4))
    bank_period_var = ctk.StringVar(value="12 mjeseci")
    state["period_days"] = 360
    state["period_start"] = None
    state["period_end"] = None
    state["unpicked_period_days"] = None
    state["unpicked_period_start"] = None
    state["unpicked_period_end"] = None

    def on_bank_period_change(choice: str):
        mapping = {
            "Svo vrijeme": None,
            "3 mjeseca": 90,
            "6 mjeseci": 180,
            "12 mjeseci": 360,
            "24 mjeseca": 720,
        }
        state["period_days"] = mapping.get(choice)
        state["period_start"] = None
        state["period_end"] = None
        refresh_dashboard()

    bank_period_menu = ctk.CTkOptionMenu(
        bank_period_frame,
        values=["Svo vrijeme", "3 mjeseca", "6 mjeseci", "12 mjeseci", "24 mjeseca"],
        variable=bank_period_var,
        command=on_bank_period_change,
    )
    bank_period_menu.pack(side="left", padx=4)

    load_baseline_lock()
    update_baseline_ui()
    update_cartice_ui()
    app.after(1000, poll_global_status)

    refresh_exchange_rate()
    sync_rate_entry()

    charts_frame = ctk.CTkFrame(tab_dashboard)
    charts_frame.pack(fill="both", expand=True, padx=10, pady=10)

    fig_customers = Figure(figsize=(4, 3), dpi=100)
    ax_customers = fig_customers.add_subplot(111)
    canvas_customers = FigureCanvasTkAgg(fig_customers, master=charts_frame)
    canvas_customers.get_tk_widget().pack(side="left", fill="both", expand=True, padx=6, pady=6)

    fig_products = Figure(figsize=(4, 3), dpi=100)
    ax_products = fig_products.add_subplot(111)
    canvas_products = FigureCanvasTkAgg(fig_products, master=charts_frame)
    canvas_products.get_tk_widget().pack(side="left", fill="both", expand=True, padx=6, pady=6)

    fig_sp_bank = Figure(figsize=(4, 3), dpi=100)
    ax_sp_bank = fig_sp_bank.add_subplot(111)
    canvas_sp_bank = FigureCanvasTkAgg(fig_sp_bank, master=charts_frame)
    canvas_sp_bank.get_tk_widget().pack(side="left", fill="both", expand=True, padx=6, pady=6)

    expenses_frame = ctk.CTkFrame(tab_troskovi)
    expenses_frame.pack(fill="both", expand=True, padx=10, pady=10)
    expenses_header = ctk.CTkFrame(expenses_frame)
    expenses_header.pack(fill="x", padx=6, pady=(0, 6))
    ctk.CTkLabel(expenses_header, text="Prikazi period:").pack(side="left", padx=(6, 4))
    expense_period_var = ctk.StringVar(value="Svo vrijeme")

    def on_expense_period_change(choice: str):
        mapping = {
            "Svo vrijeme": None,
            "3 mjeseca": 90,
            "6 mjeseci": 180,
            "12 mjeseci": 360,
            "24 mjeseca": 720,
        }
        state["expense_period_days"] = mapping.get(choice)
        state["expense_period_start"] = None
        state["expense_period_end"] = None
        refresh_expenses()

    expense_period_menu = ctk.CTkOptionMenu(
        expenses_header,
        values=["Svo vrijeme", "3 mjeseca", "6 mjeseci", "12 mjeseci", "24 mjeseca"],
        variable=expense_period_var,
        command=on_expense_period_change,
    )
    expense_period_menu.pack(side="left", padx=4)

    ctk.CTkLabel(expenses_header, text="Godina:").pack(side="left", padx=(12, 4))
    expense_year_var = ctk.StringVar(value="Sve")

    def on_expense_year_change(choice: str):
        state["expense_year"] = None if choice == "Sve" else choice
        refresh_expenses()

    expense_year_menu = ctk.CTkOptionMenu(
        expenses_header,
        values=["Sve"],
        variable=expense_year_var,
        command=on_expense_year_change,
        width=90,
    )
    expense_year_menu.pack(side="left", padx=4)

    ctk.CTkLabel(expenses_header, text="Mjesec:").pack(side="left", padx=(12, 4))
    expense_month_var = ctk.StringVar(value="Sve")

    def on_expense_month_change(choice: str):
        state["expense_month"] = None if choice == "Sve" else choice
        refresh_expenses()

    expense_month_menu = ctk.CTkOptionMenu(
        expenses_header,
        values=["Sve"] + [f"{i:02d}" for i in range(1, 13)],
        variable=expense_month_var,
        command=on_expense_month_change,
        width=70,
    )
    expense_month_menu.pack(side="left", padx=4)

    ctk.CTkLabel(expenses_header, text="Top:").pack(side="left", padx=(12, 4))
    expense_top_var = ctk.StringVar(value="5")

    def on_expense_top_change(choice: str):
        try:
            state["expense_top_n"] = int(choice)
        except ValueError:
            state["expense_top_n"] = 5
        refresh_expenses()

    expense_top_menu = ctk.CTkOptionMenu(
        expenses_header,
        values=["5", "10"],
        variable=expense_top_var,
        command=on_expense_top_change,
        width=70,
    )
    expense_top_menu.pack(side="left", padx=4)

    expense_total_var = ctk.StringVar(value="Ukupno: 0")
    ctk.CTkLabel(expenses_header, textvariable=expense_total_var).pack(side="left", padx=12)

    expenses_charts = ctk.CTkFrame(expenses_frame)
    expenses_charts.pack(fill="both", expand=True, padx=6, pady=6)

    fig_expenses_top = Figure(figsize=(4, 3), dpi=100)
    ax_expenses_top = fig_expenses_top.add_subplot(111)
    canvas_expenses_top = FigureCanvasTkAgg(fig_expenses_top, master=expenses_charts)
    canvas_expenses_top.get_tk_widget().pack(side="left", fill="both", expand=True, padx=6, pady=6)

    fig_expenses_month = Figure(figsize=(4, 3), dpi=100)
    ax_expenses_month = fig_expenses_month.add_subplot(111)
    canvas_expenses_month = FigureCanvasTkAgg(fig_expenses_month, master=expenses_charts)
    canvas_expenses_month.get_tk_widget().pack(side="left", fill="both", expand=True, padx=6, pady=6)

    returns_frame = ctk.CTkFrame(tab_povrati)
    returns_frame.pack(fill="both", expand=True, padx=10, pady=10)
    returns_ops = ctk.CTkFrame(returns_frame)
    returns_ops.pack(fill="x", padx=6, pady=(0, 6))
    ctk.CTkButton(
        returns_ops,
        text="Export povrati (detaljno)",
        command=run_export_refunds_full,
    ).pack(side="left", padx=4)

    fig_ref_customers = Figure(figsize=(4, 3), dpi=100)
    ax_ref_customers = fig_ref_customers.add_subplot(111)
    canvas_ref_customers = FigureCanvasTkAgg(fig_ref_customers, master=returns_frame)
    canvas_ref_customers.get_tk_widget().pack(side="left", fill="both", expand=True, padx=6, pady=6)

    fig_ref_items = Figure(figsize=(4, 3), dpi=100)
    ax_ref_items = fig_ref_items.add_subplot(111)
    canvas_ref_items = FigureCanvasTkAgg(fig_ref_items, master=returns_frame)
    canvas_ref_items.get_tk_widget().pack(side="left", fill="both", expand=True, padx=6, pady=6)

    fig_ref_categories = Figure(figsize=(4, 3), dpi=100)
    ax_ref_categories = fig_ref_categories.add_subplot(111)
    canvas_ref_categories = FigureCanvasTkAgg(fig_ref_categories, master=returns_frame)
    canvas_ref_categories.get_tk_widget().pack(side="left", fill="both", expand=True, padx=6, pady=6)

    nepreuzete_frame = ctk.CTkFrame(tab_nepreuzete)
    nepreuzete_frame.pack(fill="both", expand=True, padx=10, pady=10)
    nepreuzete_ops = ctk.CTkFrame(nepreuzete_frame)
    nepreuzete_ops.pack(fill="x", padx=6, pady=(0, 6))
    unpicked_period_var = ctk.StringVar(value="Svo vrijeme")
    ctk.CTkLabel(nepreuzete_ops, text="Period:").pack(side="left", padx=(6, 4))

    def on_unpicked_period_change(choice: str):
        mapping = {
            "Svo vrijeme": None,
            "3 mjeseca": 90,
            "6 mjeseci": 180,
            "12 mjeseci": 360,
            "24 mjeseca": 720,
        }
        state["unpicked_period_days"] = mapping.get(choice)
        state["unpicked_period_start"] = None
        state["unpicked_period_end"] = None
        refresh_unpicked_charts()

    unpicked_period_menu = ctk.CTkOptionMenu(
        nepreuzete_ops,
        values=["Svo vrijeme", "3 mjeseca", "6 mjeseci", "12 mjeseci", "24 mjeseca"],
        variable=unpicked_period_var,
        command=on_unpicked_period_change,
    )
    unpicked_period_menu.pack(side="left", padx=4)
    ctk.CTkLabel(nepreuzete_ops, text="Batch:").pack(side="left", padx=(12, 4))
    tracking_batch_var = ctk.StringVar(value="20")
    ent_tracking_batch = ctk.CTkEntry(nepreuzete_ops, width=60, textvariable=tracking_batch_var)
    ent_tracking_batch.pack(side="left", padx=4)
    tracking_force_var = ctk.BooleanVar(value=False)
    ctk.CTkCheckBox(nepreuzete_ops, text="Force refresh", variable=tracking_force_var).pack(
        side="left", padx=6
    )

    ctk.CTkButton(
        nepreuzete_ops,
        text="Export nepreuzete (detaljno)",
        command=run_export_unpicked_full,
    ).pack(side="left", padx=4)
    ctk.CTkButton(
        nepreuzete_ops,
        text="Dexpress analiza",
        command=run_dexpress_tracking,
    ).pack(side="left", padx=4)
    lbl_unpicked_total = ctk.CTkLabel(nepreuzete_ops, text="Nepreuzete: 0")
    lbl_unpicked_total.pack(side="left", padx=12)
    lbl_unpicked_lost = ctk.CTkLabel(nepreuzete_ops, text="Izgubljena prodaja: 0")
    lbl_unpicked_lost.pack(side="left", padx=12)
    lbl_unpicked_repeat = ctk.CTkLabel(nepreuzete_ops, text="Kupci 2+ nepreuzetih: 0")
    lbl_unpicked_repeat.pack(side="left", padx=12)
    tracking_status_var = ctk.StringVar(value="Dexpress: spremno")
    lbl_tracking_status = ctk.CTkLabel(nepreuzete_ops, textvariable=tracking_status_var)
    lbl_tracking_status.pack(side="left", padx=12)

    nepreuzete_left = ctk.CTkFrame(nepreuzete_frame)
    nepreuzete_left.pack(side="left", fill="both", expand=True, padx=6, pady=6)
    nepreuzete_right = ctk.CTkFrame(nepreuzete_frame)
    nepreuzete_right.pack(side="left", fill="y", padx=6, pady=6)
    ctk.CTkLabel(nepreuzete_right, text="Tracking summary").pack(anchor="w", padx=6, pady=(6, 2))
    txt_tracking_summary = ctk.CTkTextbox(nepreuzete_right, height=360, width=360)
    txt_tracking_summary.pack(fill="both", expand=False, padx=6, pady=(0, 6))
    ctk.CTkLabel(nepreuzete_right, text="Nepreuzete narudzbe (tracking)").pack(anchor="w", padx=6, pady=(6, 2))
    txt_nepreuzete_orders = ctk.CTkTextbox(nepreuzete_right, height=260, width=360)
    txt_nepreuzete_orders.pack(fill="both", expand=False, padx=6, pady=(0, 6))

    fig_unpicked_customers = Figure(figsize=(4, 3), dpi=100)
    ax_unpicked_customers = fig_unpicked_customers.add_subplot(111)
    canvas_unpicked_customers = FigureCanvasTkAgg(fig_unpicked_customers, master=nepreuzete_left)
    canvas_unpicked_customers.get_tk_widget().pack(side="left", fill="both", expand=True, padx=6, pady=6)

    fig_unpicked_items = Figure(figsize=(4, 3), dpi=100)
    ax_unpicked_items = fig_unpicked_items.add_subplot(111)
    canvas_unpicked_items = FigureCanvasTkAgg(fig_unpicked_items, master=nepreuzete_left)
    canvas_unpicked_items.get_tk_widget().pack(side="left", fill="both", expand=True, padx=6, pady=6)

    test_frame = ctk.CTkFrame(tab_test)
    test_frame.pack(fill="both", expand=True, padx=10, pady=10)
    test_top = ctk.CTkFrame(test_frame)
    test_top.pack(fill="x", padx=6, pady=(0, 6))
    ctk.CTkLabel(test_top, text="Session size:").pack(side="left", padx=(6, 4))
    test_size_var = ctk.StringVar(value="30")
    ent_test_size = ctk.CTkEntry(test_top, width=60, textvariable=test_size_var)
    ent_test_size.pack(side="left", padx=4)
    lbl_test_status = ctk.CTkLabel(test_top, text="Spremno.")
    lbl_test_status.pack(side="left", padx=12)

    test_body = ctk.CTkFrame(test_frame)
    test_body.pack(fill="both", expand=True, padx=6, pady=6)
    test_left = ctk.CTkFrame(test_body)
    test_left.pack(side="left", fill="both", expand=True, padx=6, pady=6)
    test_right = ctk.CTkFrame(test_body)
    test_right.pack(side="left", fill="both", expand=True, padx=6, pady=6)

    ctk.CTkLabel(test_left, text="SP narudzba").pack(anchor="w", padx=6, pady=(6, 2))
    txt_test_order = ctk.CTkTextbox(test_left, height=220)
    txt_test_order.pack(fill="both", expand=True, padx=6, pady=(0, 6))
    ctk.CTkLabel(test_right, text="Minimax racun").pack(anchor="w", padx=6, pady=(6, 2))
    txt_test_invoice = ctk.CTkTextbox(test_right, height=220)
    txt_test_invoice.pack(fill="both", expand=True, padx=6, pady=(0, 6))

    test_actions = ctk.CTkFrame(test_frame)
    test_actions.pack(fill="x", padx=6, pady=(0, 6))
    btn_test_suspicious = ctk.CTkButton(test_actions, text="Provjeri sumnjive")
    btn_test_suspicious.pack(side="left", padx=4)
    btn_test_all = ctk.CTkButton(test_actions, text="Provjeri sve")
    btn_test_all.pack(side="left", padx=4)
    btn_test_confirm = ctk.CTkButton(test_actions, text="Potvrdi")
    btn_test_confirm.pack(side="left", padx=4)
    btn_test_reject = ctk.CTkButton(test_actions, text="Odbij")
    btn_test_reject.pack(side="left", padx=4)
    btn_test_skip = ctk.CTkButton(test_actions, text="Preskoci")
    btn_test_skip.pack(side="left", padx=4)
    btn_test_stop = ctk.CTkButton(test_actions, text="Prekini sesiju")
    btn_test_stop.pack(side="left", padx=4)

    btn_test_suspicious.configure(command=lambda: start_test_session("review"))
    btn_test_all.configure(command=lambda: start_test_session("all"))
    btn_test_confirm.configure(command=lambda: _record_decision("potvrdi"))
    btn_test_reject.configure(command=lambda: _record_decision("odbij"))
    btn_test_skip.configure(command=lambda: _record_decision("preskoci"))
    btn_test_stop.configure(command=stop_test_session)

    poslovanje_body = ctk.CTkFrame(tab_poslovanje)
    poslovanje_body.pack(fill="both", expand=True, padx=10, pady=10)
    poslovanje_body.grid_columnconfigure(0, weight=0)
    poslovanje_body.grid_columnconfigure(1, weight=1)
    poslovanje_body.grid_rowconfigure(0, weight=1)

    ops_frame = ctk.CTkFrame(poslovanje_body)
    ops_frame.grid(row=0, column=0, sticky="nsw", padx=(0, 10), pady=0)

    uvoz_frame = ctk.CTkFrame(ops_frame)
    uvoz_frame.pack(anchor="nw", fill="x", pady=(0, 10))
    kalkulacije_dir_var = ctk.StringVar(value=state["kalkulacije_output_dir"])
    kartice_dir_var = ctk.StringVar(value=state["kartice_output_dir"])
    ctk.CTkLabel(uvoz_frame, text="Uvoz novih fajlova").pack(anchor="w", pady=(0, 2))
    ctk.CTkLabel(
        uvoz_frame,
        text="Workflow: 1) SP Narudzbe 2) Minimax 3) SP Uplate 4) Banka XML 5) SP Preuzimanja",
    ).pack(anchor="w", pady=(0, 6))
    ctk.CTkButton(
        uvoz_frame,
        text="Provjeri zadnje",
        command=show_last_imports,
        width=140,
        height=24,
        fg_color="#dc3545",
        hover_color="#c82333",
    ).pack(anchor="w", pady=(0, 6))
    ctk.CTkButton(uvoz_frame, text="SP Narudzbe", command=lambda: run_import_folder(import_sp_orders, "SP Narudzbe (folder)", "*.xlsx")).pack(anchor="w", pady=2)
    ctk.CTkButton(uvoz_frame, text="Minimax", command=lambda: run_import_folder(import_minimax, "Minimax (folder)", "*.xlsx")).pack(anchor="w", pady=2)
    ctk.CTkButton(uvoz_frame, text="SP Uplate", command=lambda: run_import_folder(import_sp_payments, "SP Uplate (folder)", "*.xlsx")).pack(anchor="w", pady=2)
    ctk.CTkButton(uvoz_frame, text="Banka XML", command=lambda: run_import_folder(import_bank_xml, "Banka XML (folder)", "*.xml")).pack(anchor="w", pady=2)
    ctk.CTkButton(uvoz_frame, text="SP Preuzimanja", command=lambda: run_import_folder(import_sp_returns, "SP Preuzimanja (folder)", "*.xlsx")).pack(anchor="w", pady=2)
    ctk.CTkFrame(uvoz_frame, height=1, fg_color="#7a7a7a").pack(fill="x", pady=(8, 6))
    ctk.CTkLabel(uvoz_frame, text="Kalkulacije i kartice artikala").pack(anchor="w", pady=(0, 4))
    kalkulacije_frame = ctk.CTkFrame(uvoz_frame)
    kalkulacije_frame.pack(fill="x", pady=(0, 4))
    ctk.CTkButton(
        kalkulacije_frame,
        text="Uvezi nove kalkulacije",
        command=import_kalkulacije_flow,
    ).pack(side="left")
    ctk.CTkEntry(
        kalkulacije_frame,
        textvariable=kalkulacije_dir_var,
        placeholder_text="Folder za CSV fajlove kalkulacija",
    ).pack(side="left", fill="x", expand=True, padx=(6, 2))
    ctk.CTkButton(
        kalkulacije_frame,
        text="Folder",
        width=90,
        command=lambda: _choose_output_dir(
            "kalkulacije_output_dir",
            kalkulacije_dir_var,
            "Folder za kalkulacije CSV",
        ),
    ).pack(side="left")
    kartice_frame = ctk.CTkFrame(uvoz_frame)
    kartice_frame.pack(fill="x", pady=(0, 4))
    ctk.CTkButton(
        kartice_frame,
        text="Uvezi kartice artikala",
        command=import_kartice_flow,
    ).pack(side="left")
    ctk.CTkEntry(
        kartice_frame,
        textvariable=kartice_dir_var,
        placeholder_text="Folder za CSV fajlove kartica",
    ).pack(side="left", fill="x", expand=True, padx=(6, 2))
    ctk.CTkButton(
        kartice_frame,
        text="Folder",
        width=90,
        command=lambda: _choose_output_dir(
            "kartice_output_dir",
            kartice_dir_var,
            "Folder za kartice CSV",
        ),
    ).pack(side="left")

    akcije_frame = ctk.CTkFrame(ops_frame)
    akcije_frame.pack(anchor="nw", fill="x")
    ctk.CTkLabel(akcije_frame, text="Akcije").pack(anchor="w", pady=(0, 6))
    btn_match_minimax = ctk.CTkButton(
        akcije_frame,
        text="Match Minimax",
        command=lambda: run_action_async_process(
            run_match_minimax_process,
            [str(state["db_path"])],
            "Match Minimax",
            progress_task="match_minimax",
        ),
    )
    btn_match_minimax.pack(anchor="w", pady=2)
    btn_match_banka = ctk.CTkButton(
        akcije_frame,
        text="Match Banka",
        command=lambda: run_action_async_process(
            run_match_bank_process,
            [str(state["db_path"]), 2],
            "Match Banka",
            progress_task="match_bank",
        ),
    )
    btn_match_banka.pack(anchor="w", pady=2)
    btn_close = ctk.CTkButton(akcije_frame, text="Zatvori racune", command=lambda: run_action(close_invoices_from_confirmed_matches))
    btn_close.pack(anchor="w", pady=2)
    btn_reset_matches = ctk.CTkButton(akcije_frame, text="Reset match (Minimax)", command=run_reset_minimax_matches)
    btn_reset_matches.pack(anchor="w", pady=2)
    btn_export = ctk.CTkButton(akcije_frame, text="Export (osnovno)", command=run_export_basic)
    btn_export.pack(anchor="w", pady=6)
    btn_export_unmatched = ctk.CTkButton(
        akcije_frame,
        text="Export neuparene",
        command=lambda: run_export_single(report_unmatched_reasons, "unmatched-reasons"),
    )
    btn_export_unmatched.pack(anchor="w", pady=2)
    btn_export_refunds = ctk.CTkButton(
        akcije_frame,
        text="Export povrati (banka)",
        command=run_export_bank_refunds,
    )
    btn_export_refunds.pack(anchor="w", pady=2)
    btn_open_exports = ctk.CTkButton(akcije_frame, text="Otvori exports", command=open_exports)
    btn_open_exports.pack(anchor="w", pady=2)
    btn_refresh_lists = ctk.CTkButton(akcije_frame, text="Osvjezi liste", command=refresh_poslovanje_lists)
    btn_refresh_lists.pack(anchor="w", pady=2)

    status_var = ctk.StringVar(value="Spremno.")
    ctk.CTkLabel(ops_frame, textvariable=status_var).pack(anchor="w", pady=(8, 0))
    progress = ctk.CTkProgressBar(ops_frame, mode="indeterminate")
    progress.pack(anchor="w", fill="x", pady=(4, 0))
    progress_pct_var = ctk.StringVar(value="")
    ctk.CTkLabel(ops_frame, textvariable=progress_pct_var).pack(anchor="w", pady=(2, 0))
    progress_eta_var = ctk.StringVar(value="")
    ctk.CTkLabel(ops_frame, textvariable=progress_eta_var).pack(anchor="w", pady=(2, 0))

    action_buttons = [
        btn_match_minimax,
        btn_match_banka,
        btn_close,
        btn_reset_matches,
        btn_export,
        btn_export_refunds,
        btn_open_exports,
        btn_refresh_lists,
    ]

    update_baseline_ui()

    executor = concurrent.futures.ProcessPoolExecutor(max_workers=1)

    lists_frame = ctk.CTkFrame(poslovanje_body)
    lists_frame.grid(row=0, column=1, sticky="nsew")

    left_list = ctk.CTkFrame(lists_frame)
    left_list.pack(side="left", fill="both", expand=True, padx=6, pady=6)
    ctk.CTkLabel(left_list, text="Fali Minimax racun").pack(anchor="w", padx=6, pady=(6, 2))

    txt_needs = ctk.CTkTextbox(left_list, height=200)
    txt_needs.pack(fill="both", expand=True, padx=6, pady=6)
    txt_needs.configure(state="disabled")

    right_list = ctk.CTkFrame(lists_frame)
    right_list.pack(side="left", fill="both", expand=True, padx=6, pady=6)
    ctk.CTkLabel(right_list, text="Neuparene narudzbe").pack(anchor="w", padx=6, pady=(6, 2))
    txt_unmatched = ctk.CTkTextbox(right_list, height=200)
    txt_unmatched.pack(fill="both", expand=True, padx=6, pady=6)
    txt_unmatched.configure(state="disabled")

    refresh_dashboard()
    refresh_poslovanje_lists()
    app.mainloop()

def run_smoke_tests() -> int:
    failures = 0

    def check(label: str, cond: bool, detail: str = ""):
        nonlocal failures
        if cond:
            print(f"OK: {label}")
        else:
            failures += 1
            print(f"FAIL: {label} {detail}")

    check("normalize_phone +381", normalize_phone("+381641234567") == "0641234567")
    check("normalize_phone 064", normalize_phone("064 123 4567") == "0641234567")
    check("normalize_phone 6xx", normalize_phone("61234567").startswith("0"))

    key = compute_customer_key("0641234567", None, "Test User", "Beograd")
    check("customer_key phone", key.startswith("phone:"))

    history = [
        {"statusTime": "21.11.2025 18:33:17", "statusValue": "Pošiljka je preuzeta od pošiljaoca"},
        {"statusTime": "24.11.2025 07:51:56", "statusValue": "Pošiljka zadužena za isporuku"},
        {"statusTime": "25.11.2025 13:31:31", "statusValue": "Ponovni pokušaj isporuke, nema nikoga na adresi"},
        {"statusTime": "08.12.2025 11:40:58", "statusValue": "Pošiljka je vraćena pošiljaocu"},
    ]
    events, summary = analyze_tracking_history(history)
    check("tracking received_at", summary.get("received_at") is not None)
    check("tracking first_out_for_delivery_at", summary.get("first_out_for_delivery_at") is not None)
    check("tracking has_returned", summary.get("has_returned") == 1)
    check("tracking attempts >=1", (summary.get("delivery_attempts") or 0) >= 1)

    print(f"Tests finished. Failures: {failures}")
    return failures

def main() -> None:
    parser = argparse.ArgumentParser(description="SRB1.0 import tool")
    parser.add_argument("--db", type=Path, default=DB_PATH)
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("init-db")

    sp_orders = sub.add_parser("import-sp-orders")
    sp_orders.add_argument("path", type=Path)

    sp_payments = sub.add_parser("import-sp-payments")
    sp_payments.add_argument("path", type=Path)

    sp_returns = sub.add_parser("import-sp-returns")
    sp_returns.add_argument("path", type=Path)

    minimax = sub.add_parser("import-minimax")
    minimax.add_argument("path", type=Path)

    minimax_items = sub.add_parser("import-minimax-items")
    minimax_items.add_argument("path", type=Path)

    bank = sub.add_parser("import-bank-xml")
    bank.add_argument("path", type=Path)

    bank_match = sub.add_parser("match-bank")
    bank_match.add_argument("--day-tolerance", type=int, default=2)

    sub.add_parser("extract-bank-refunds")

    match = sub.add_parser("match-minimax")
    match.add_argument("--auto-threshold", type=int, default=70)
    match.add_argument("--review-threshold", type=int, default=50)

    review = sub.add_parser("list-review")
    confirm = sub.add_parser("confirm-match")
    confirm.add_argument("match_id", type=int)

    close_invoices = sub.add_parser("close-invoices")

    report = sub.add_parser("report")
    report.add_argument(
        "name",
        choices=[
            "unmatched",
            "unmatched-reasons",
            "conflicts",
            "nearest",
            "open",
            "returns",
            "needs-invoice",
            "needs-invoice-orders",
            "no-value",
            "candidates",
            "unmatched-candidates",
            "unmatched-candidates-grouped",
            "storno",
            "bank-sp",
            "bank-refunds",
            "bank-refunds-extracted",
            "bank-unmatched-sp",
            "bank-unmatched-refunds",
            "sp-vs-bank",
            "alarms",
            "refunds-no-storno",
            "order-amount-issues",
            "duplicate-customers",
            "top-customers",
            "category-sales",
            "category-returns",
            "minimax-items",
        ],
        help="unmatched | unmatched-reasons | conflicts | nearest | open | returns | needs-invoice | needs-invoice-orders | no-value | candidates | unmatched-candidates | unmatched-candidates-grouped | storno | bank-sp | bank-refunds | bank-refunds-extracted | bank-unmatched-sp | bank-unmatched-refunds | sp-vs-bank | alarms | refunds-no-storno | order-amount-issues | duplicate-customers | top-customers | category-sales | category-returns | minimax-items",
    )
    report.add_argument(
        "--out-dir",
        type=Path,
        default=Path("exports"),
        help="Folder for CSV/XLSX exports",
    )
    report.add_argument(
        "--format",
        choices=["csv", "xlsx"],
        default="csv",
        help="Export format",
    )
    report.add_argument(
        "--days",
        type=int,
        default=7,
        help="Days threshold for alarms report",
    )

    export_review = sub.add_parser("export-review")
    export_review.add_argument(
        "--out-dir",
        type=Path,
        default=Path("exports"),
        help="Folder for CSV/XLSX exports",
    )
    export_review.add_argument(
        "--format",
        choices=["csv", "xlsx"],
        default="csv",
        help="Export format",
    )

    export_all = sub.add_parser("export-all")
    export_all.add_argument(
        "--out-dir",
        type=Path,
        default=Path("exports"),
        help="Folder for CSV/XLSX exports",
    )
    export_all.add_argument(
        "--format",
        choices=["csv", "xlsx"],
        default="csv",
        help="Export format",
    )

    import_confirm = sub.add_parser("import-confirm")
    import_confirm.add_argument("path", type=Path, help="CSV/XLSX sa match_id i confirm")

    ui = sub.add_parser("ui")
    ui.add_argument("--db", type=Path, default=DB_PATH)

    sub.add_parser("tests")

    cat = sub.add_parser("category")
    cat_sub = cat.add_subparsers(dest="action", required=True)
    cat_prefix = cat_sub.add_parser("add-prefix")
    cat_prefix.add_argument("prefix")
    cat_prefix.add_argument("name")
    cat_override = cat_sub.add_parser("add-sku-override")
    cat_override.add_argument("sku")
    cat_override.add_argument("category")
    cat_custom = cat_sub.add_parser("add-custom-sku")
    cat_custom.add_argument("sku")

    args = parser.parse_args()
    conn = connect_db(args.db)
    init_db(conn)
    ensure_customer_keys(conn)

    if not args.cmd:
        run_ui(args.db)
        return
    if args.cmd == "init-db":
        return
    if args.cmd == "tests":
        conn.close()
        run_smoke_tests()
        return
    if args.cmd == "import-sp-orders":
        import_sp_orders(conn, args.path)
    elif args.cmd == "import-sp-payments":
        import_sp_payments(conn, args.path)
    elif args.cmd == "import-sp-returns":
        import_sp_returns(conn, args.path)
    elif args.cmd == "import-minimax":
        import_minimax(conn, args.path)
    elif args.cmd == "import-minimax-items":
        import_minimax_items(conn, args.path)
    elif args.cmd == "import-bank-xml":
        import_bank_xml(conn, args.path)
    elif args.cmd == "match-bank":
        match_bank_sp_payments(conn, args.day_tolerance)
        match_bank_refunds(conn)
    elif args.cmd == "extract-bank-refunds":
        extract_bank_refunds(conn)
    elif args.cmd == "match-minimax":
        match_minimax(conn, args.auto_threshold, args.review_threshold)
    elif args.cmd == "list-review":
        list_review_matches(conn)
    elif args.cmd == "confirm-match":
        confirm_match(conn, args.match_id)
    elif args.cmd == "close-invoices":
        close_invoices_from_confirmed_matches(conn)
    elif args.cmd == "report":
        if args.name == "unmatched":
            cols, rows = report_unmatched_orders(conn, return_rows=True)
        elif args.name == "unmatched-reasons":
            cols, rows = report_unmatched_reasons(conn, return_rows=True)
        elif args.name == "conflicts":
            cols, rows = report_conflicts(conn, return_rows=True)
        elif args.name == "nearest":
            cols, rows = report_nearest_invoice(conn, return_rows=True)
        elif args.name == "open":
            cols, rows = report_open_invoices(conn, return_rows=True)
        elif args.name == "returns":
            cols, rows = report_returns(conn, return_rows=True)
        elif args.name == "needs-invoice":
            cols, rows = report_needs_invoice(conn, return_rows=True)
        elif args.name == "needs-invoice-orders":
            cols, rows = report_needs_invoice_orders(conn, return_rows=True)
        elif args.name == "no-value":
            cols, rows = report_no_value_orders(conn, return_rows=True)
        elif args.name == "candidates":
            cols, rows = report_candidates(conn, return_rows=True)
        elif args.name == "unmatched-candidates":
            cols, rows = report_unmatched_with_candidates(conn, return_rows=True)
        elif args.name == "unmatched-candidates-grouped":
            cols, rows = report_unmatched_with_candidates_grouped(conn, return_rows=True)
        elif args.name == "storno":
            cols, rows = report_storno(conn, return_rows=True)
        elif args.name == "bank-sp":
            cols, rows = report_bank_sp(conn, return_rows=True)
        elif args.name == "bank-refunds":
            cols, rows = report_bank_refunds(conn, return_rows=True)
        elif args.name == "bank-refunds-extracted":
            cols, rows = report_bank_refunds_extracted(conn, return_rows=True)
        elif args.name == "bank-unmatched-sp":
            cols, rows = report_bank_unmatched_sp(conn, return_rows=True)
        elif args.name == "bank-unmatched-refunds":
            cols, rows = report_bank_unmatched_refunds(conn, return_rows=True)
        elif args.name == "sp-vs-bank":
            cols, rows = report_sp_vs_bank(conn, return_rows=True)
        elif args.name == "alarms":
            cols, rows = report_alarms(conn, days=args.days, return_rows=True)
        elif args.name == "refunds-no-storno":
            cols, rows = report_refunds_without_storno(conn, return_rows=True)
        elif args.name == "order-amount-issues":
            cols, rows = report_order_amount_issues(conn, return_rows=True)
        elif args.name == "duplicate-customers":
            cols, rows = report_duplicate_customers(conn, return_rows=True)
        elif args.name == "top-customers":
            cols, rows = report_top_customers(conn, return_rows=True)
        elif args.name == "category-sales":
            cols, rows = report_category_sales(conn, return_rows=True)
        elif args.name == "category-returns":
            cols, rows = report_category_returns(conn, return_rows=True)
        elif args.name == "minimax-items":
            cols, rows = report_minimax_items(conn, return_rows=True)
        out_path = write_report(cols, rows, args.out_dir, args.name, args.format)
        print(f"Export snimljen u: {out_path}")
    elif args.cmd == "export-review":
        cols, rows = list_review_matches(conn, return_rows=True)
        out_path = write_report(cols, rows, args.out_dir, "review", args.format)
        print(f"Export snimljen u: {out_path}")
    elif args.cmd == "export-all":
        exports = []
        cols, rows = report_unmatched_orders(conn, return_rows=True)
        exports.append(write_report(cols, rows, args.out_dir, "unmatched", args.format))
        cols, rows = report_open_invoices(conn, return_rows=True)
        exports.append(write_report(cols, rows, args.out_dir, "open", args.format))
        cols, rows = report_returns(conn, return_rows=True)
        exports.append(write_report(cols, rows, args.out_dir, "returns", args.format))
        cols, rows = report_needs_invoice(conn, return_rows=True)
        exports.append(write_report(cols, rows, args.out_dir, "needs-invoice", args.format))
        cols, rows = report_needs_invoice_orders(conn, return_rows=True)
        exports.append(write_report(cols, rows, args.out_dir, "needs-invoice-orders", args.format))
        cols, rows = report_no_value_orders(conn, return_rows=True)
        exports.append(write_report(cols, rows, args.out_dir, "no-value", args.format))
        cols, rows = report_candidates(conn, return_rows=True)
        exports.append(write_report(cols, rows, args.out_dir, "candidates", args.format))
        cols, rows = report_unmatched_with_candidates(conn, return_rows=True)
        exports.append(write_report(cols, rows, args.out_dir, "unmatched-candidates", args.format))
        cols, rows = report_unmatched_with_candidates_grouped(conn, return_rows=True)
        exports.append(write_report(cols, rows, args.out_dir, "unmatched-candidates-grouped", args.format))
        cols, rows = report_storno(conn, return_rows=True)
        exports.append(write_report(cols, rows, args.out_dir, "storno", args.format))
        cols, rows = report_sp_vs_bank(conn, return_rows=True)
        exports.append(write_report(cols, rows, args.out_dir, "sp-vs-bank", args.format))
        cols, rows = report_refunds_without_storno(conn, return_rows=True)
        exports.append(write_report(cols, rows, args.out_dir, "refunds-no-storno", args.format))
        cols, rows = report_alarms(conn, days=7, return_rows=True)
        exports.append(write_report(cols, rows, args.out_dir, "alarms", args.format))
        cols, rows = report_order_amount_issues(conn, return_rows=True)
        exports.append(write_report(cols, rows, args.out_dir, "order-amount-issues", args.format))
        cols, rows = report_duplicate_customers(conn, return_rows=True)
        exports.append(write_report(cols, rows, args.out_dir, "duplicate-customers", args.format))
        cols, rows = report_top_customers(conn, return_rows=True)
        exports.append(write_report(cols, rows, args.out_dir, "top-customers", args.format))
        cols, rows = report_category_sales(conn, return_rows=True)
        exports.append(write_report(cols, rows, args.out_dir, "category-sales", args.format))
        cols, rows = report_category_returns(conn, return_rows=True)
        exports.append(write_report(cols, rows, args.out_dir, "category-returns", args.format))
        cols, rows = report_minimax_items(conn, return_rows=True)
        exports.append(write_report(cols, rows, args.out_dir, "minimax-items", args.format))
        cols, rows = list_review_matches(conn, return_rows=True)
        exports.append(write_report(cols, rows, args.out_dir, "review", args.format))
        print("Exporti snimljeni:")
        for path in exports:
            print(path)
    elif args.cmd == "import-confirm":
        apply_review_decisions(conn, args.path)
    elif args.cmd == "ui":
        run_ui(args.db)
    elif args.cmd == "category":
        if args.action == "add-prefix":
            add_category_prefix(args.prefix, args.name)
        elif args.action == "add-sku-override":
            add_sku_category_override(args.sku, args.category)
        elif args.action == "add-custom-sku":
            add_custom_sku(args.sku)


if __name__ == "__main__":
    main()
