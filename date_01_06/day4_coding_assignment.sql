-- Create schema
CREATE SCHEMA IF NOT EXISTS shopey;

CREATE TABLE shopey.orders (
    order_id    SERIAL PRIMARY KEY,
    customer_id INT NOT NULL REFERENCES shopey.customers(customer_id),
    order_date  TIMESTAMP DEFAULT NOW(),
    status      VARCHAR(20) CHECK (status IN ('Pending','Confirmed','Shipped','Delivered','Cancelled'))
                DEFAULT 'Pending'
);

-- Order Lines
CREATE TABLE shopey.order_lines (
    line_id     SERIAL PRIMARY KEY,
    order_id    INT NOT NULL REFERENCES shopey.orders(order_id),
    product_id  INT NOT NULL REFERENCES shopey.products(product_id),
    quantity    INT NOT NULL CHECK (quantity > 0),
    unit_price  NUMERIC(10,2) NOT NULL
);

-- Payments
CREATE TABLE shopey.payments (
    payment_id   SERIAL PRIMARY KEY,
    order_id     INT UNIQUE NOT NULL REFERENCES shopey.orders(order_id),
    payment_date TIMESTAMP,
    method       VARCHAR(50) CHECK (method IN ('Card','PayPal','Bank Transfer','Wallet')),
    amount       NUMERIC(10,2) NOT NULL,
    status       VARCHAR(20) CHECK (status IN ('Pending','Paid','Failed','Refunded')) DEFAULT 'Pending'
);

SELECT
    c.first_name || ' ' || c.last_name      AS customer_name,
    COUNT(DISTINCT o.order_id)               AS total_orders,
    SUM(ol.quantity * ol.unit_price)         AS total_spent,
    RANK() OVER (
        ORDER BY SUM(ol.quantity * ol.unit_price) DESC
    )                                        AS customer_rank
FROM shopey.customers c
JOIN shopey.orders     o  ON c.customer_id = o.customer_id
JOIN shopey.order_lines ol ON o.order_id   = ol.order_id
GROUP BY c.customer_id, c.first_name, c.last_name
HAVING COUNT(DISTINCT o.order_id) >= 1
ORDER BY customer_rank ASC;