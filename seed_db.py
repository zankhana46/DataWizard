import sqlite3
import random
from datetime import date, timedelta
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "data", "sample.db")

REGIONS = ["North", "South", "East", "West"]
PRODUCTS = ["Laptop", "Phone", "Tablet", "Monitor", "Keyboard", "Mouse"]
PLANS = ["free", "pro", "enterprise"]

def random_date(start: date, end: date) -> str:
    delta = (end - start).days
    return (start + timedelta(days=random.randint(0, delta))).isoformat()

def seed():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()

    cur.executescript("""
        DROP TABLE IF EXISTS sales;
        DROP TABLE IF EXISTS customers;
        DROP TABLE IF EXISTS monthly_targets;

        CREATE TABLE sales (
            id INTEGER PRIMARY KEY,
            date TEXT,
            region TEXT,
            product TEXT,
            quantity INTEGER,
            revenue REAL
        );

        CREATE TABLE customers (
            id INTEGER PRIMARY KEY,
            name TEXT,
            region TEXT,
            signup_date TEXT,
            plan TEXT
        );

        CREATE TABLE monthly_targets (
            region TEXT,
            month TEXT,
            target_revenue REAL,
            PRIMARY KEY (region, month)
        );
    """)

    # sales — 200 rows
    price_map = {
        "Laptop": 1200, "Phone": 800, "Tablet": 500,
        "Monitor": 350, "Keyboard": 120, "Mouse": 60,
    }
    sales_rows = []
    for i in range(1, 201):
        region = random.choice(REGIONS)
        product = random.choice(PRODUCTS)
        qty = random.randint(1, 10)
        base_price = price_map[product]
        revenue = round(qty * base_price * random.uniform(0.85, 1.15), 2)
        d = random_date(date(2024, 1, 1), date(2024, 12, 31))
        sales_rows.append((i, d, region, product, qty, revenue))

    cur.executemany(
        "INSERT INTO sales VALUES (?,?,?,?,?,?)", sales_rows
    )

    # customers — 100 rows
    first_names = ["Alice", "Bob", "Carol", "David", "Eva", "Frank", "Grace",
                   "Hank", "Iris", "Jack", "Karen", "Leo", "Mia", "Nora", "Oscar"]
    last_names = ["Smith", "Jones", "Brown", "Taylor", "Wilson", "Davis",
                  "Clark", "Hall", "Lewis", "Walker"]
    customer_rows = []
    for i in range(1, 101):
        name = f"{random.choice(first_names)} {random.choice(last_names)}"
        region = random.choice(REGIONS)
        signup = random_date(date(2022, 1, 1), date(2024, 12, 31))
        plan = random.choices(PLANS, weights=[50, 35, 15])[0]
        customer_rows.append((i, name, region, signup, plan))

    cur.executemany(
        "INSERT INTO customers VALUES (?,?,?,?,?)", customer_rows
    )

    # monthly_targets — 4 regions × 12 months
    target_rows = []
    for region in REGIONS:
        for month in range(1, 13):
            month_str = f"2024-{month:02d}"
            target = round(random.uniform(15000, 45000), 2)
            target_rows.append((region, month_str, target))

    cur.executemany(
        "INSERT INTO monthly_targets VALUES (?,?,?)", target_rows
    )

    con.commit()
    con.close()
    print(f"Database seeded at {DB_PATH}")
    print("  sales: 200 rows")
    print("  customers: 100 rows")
    print("  monthly_targets: 48 rows")

if __name__ == "__main__":
    seed()
