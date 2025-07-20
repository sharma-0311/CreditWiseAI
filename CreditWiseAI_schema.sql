-- Create the database (optional)
CREATE DATABASE IF NOT EXISTS CreditWiseAI;
USE CreditWiseAI;

-- Table: applicant_inputs
CREATE TABLE IF NOT EXISTS applicant_inputs (
    id INT AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(100) DEFAULT 'Anonymous',
    age INT,
    credit_amount FLOAT,
    employment_duration VARCHAR(100),
    savings_status VARCHAR(100),
    loan_purpose VARCHAR(100),
    housing VARCHAR(50),
    number_of_dependents INT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Table: prediction_logs
CREATE TABLE IF NOT EXISTS prediction_logs (
    id INT AUTO_INCREMENT PRIMARY KEY,
    applicant_id INT,
    is_creditworthy BOOLEAN,
    confidence_score FLOAT,
    prediction_time DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (applicant_id) REFERENCES applicant_inputs(id)
);
