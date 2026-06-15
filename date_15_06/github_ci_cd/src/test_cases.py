"""FinSight AI — canonical 20-case test set."""

TEST_CASES = [
    # EASY
    {"id":"TC01","difficulty":"easy",
     "data":"Borrower: Apex Manufacturing Ltd. Revenue: $12.4M. EBITDA: $2.1M. Debt: $4.5M. DSCR: 1.8x. Loan request: $1.5M term loan, 5yr."},
    {"id":"TC02","difficulty":"easy",
     "data":"Borrower: GreenLeaf Organics Inc. Revenue: $5.2M. Net Profit: $420K. Current ratio: 2.1. No debt. Loan request: $800K equipment."},
    {"id":"TC03","difficulty":"easy",
     "data":"Borrower: Sunrise Hotels Group. Revenue: $18.7M. EBITDA: $3.9M. Existing debt: $7.2M. LTV: 62%. Loan request: $2M expansion."},
    {"id":"TC04","difficulty":"easy",
     "data":"Borrower: TechBridge Solutions LLC. ARR $3.1M. MRR growth 12% QoQ. Churn 2.1%. No debt. Loan request: $500K working capital."},
    {"id":"TC05","difficulty":"easy",
     "data":"Borrower: Coastal Fisheries Co. Revenue: $9.8M. EBITDA: $1.4M. DSCR: 1.6x. Fleet value $4.2M. Loan request: $1.2M."},
    # MEDIUM
    {"id":"TC06","difficulty":"medium",
     "data":"Borrower: RetailPro Chain. Revenue $22M (down 8% YoY). EBITDA $1.1M. Leases $4.8M/yr. DSCR 1.05x. Loan request: $3M."},
    {"id":"TC07","difficulty":"medium",
     "data":"Borrower: NovaBio Pharma. Pre-revenue. Burn $200K/mo. Runway 14mo. VC $4M raised. Loan request: $1.5M bridge."},
    {"id":"TC08","difficulty":"medium",
     "data":"Borrower: Atlas Construction. Revenue $31M. Gross margin 8%. DSCR 1.3x. Litigation $500K unresolved. Loan request: $4M."},
    {"id":"TC09","difficulty":"medium",
     "data":"Borrower: PrimeAgri Partners. Revenue $7.4M. EBITDA $900K. Farmland $3.5M. Drought region. Loan request: $1.8M."},
    {"id":"TC10","difficulty":"medium",
     "data":"Borrower: Urban Mobility Inc. Revenue $1.2M. Burn $350K/mo. Negative EBITDA. City contract $5M/3yr. Loan request: $600K."},
    # HARD
    {"id":"TC11","difficulty":"hard",
     "data":"Borrower: GlobalTrade. Revenue $45M (70% one customer). DSCR 2.1x. Customer restructuring announced. Loan request: $6M."},
    {"id":"TC12","difficulty":"hard",
     "data":"Borrower: DataVault Systems. Revenue $8M. EBITDA reported $2.4M. Related-party txn $1.8M. Adjusted EBITDA $600K. Loan: $2.5M."},
    {"id":"TC13","difficulty":"hard",
     "data":"Borrower: Heritage Real Estate Fund. NAV $28M. Leverage 3.2x. LTV 71%. ICR 1.1x. Two assets negative watch. Loan: $8M."},
    {"id":"TC14","difficulty":"hard",
     "data":"Borrower: CryptoAsset Ventures. Revenue $3.1M crypto trading. No audited financials. Assets $4.2M digital. Loan request: $1M."},
    {"id":"TC15","difficulty":"hard",
     "data":"Borrower: MedDevice International. Revenue $14M. Profitable. 3 jurisdictions, pending regulatory review. Recall risk 60% revenue. Loan: $3.5M."},
    # ADVERSARIAL
    {"id":"TC16","difficulty":"adversarial",
     "data":"Borrower: Pinnacle Energy. Revenue $19M. Net income $3.2M. Tax 25% but taxes paid $250K (expect $800K). EBITDA $5.4M no D&A. Loan: $4M."},
    {"id":"TC17","difficulty":"adversarial",
     "data":"Borrower: FreshFoods Co. Revenue $11M Q1-annualised. Q1 is 35% of annual => actual ~$7M. Q1 submitted as representative. Loan: $2.5M."},
    {"id":"TC18","difficulty":"adversarial",
     "data":"Borrower: TechStart Alpha. ARR $2.4M. MRR $300K. $300K*12=$3.6M≠$2.4M. ARPU implied $67K, stated $15K. Loan: $1M."},
    {"id":"TC19","difficulty":"adversarial",
     "data":"Borrower: LuxProperty Group. Properties appraised $24M. Debt $18M. LTV stated 65% actual 75%. Stressed ICR +200bps: 0.85x. Loan: $3M."},
    {"id":"TC20","difficulty":"adversarial",
     "data":"Borrower: NovaMed Clinic Group. Revenue $6.2M. EBITDA $1.8M (29% margin, typical 12-15%). No breakdown. No audited accounts. Loan: $2M."},
]

# Smoke test — 5 easy cases only, used in CI for speed
SMOKE_TEST_CASES = [t for t in TEST_CASES if t["difficulty"] == "easy"]
