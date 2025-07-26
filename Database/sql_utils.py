from db_config import connect_db

def insert_applicant(data):
    conn = connect_db()
    cursor = conn.cursor()

    query = """
        INSERT INTO applicant_inputs (
            name, age, credit_amount, employment_duration, savings_status,
            loan_purpose, housing, number_of_dependents
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    """

    values = (
        data['name'],
        data['age'],
        data['credit_amount'],
        data['employment_duration'],
        data['savings_status'],
        data['loan_purpose'],
        data['housing'],
        data['number_of_dependents']
    )

    cursor.execute(query, values)
    conn.commit()
    applicant_id = cursor.lastrowid
    conn.close()
    return applicant_id

def insert_prediction(applicant_id, is_creditworthy, confidence):
    conn = connect_db()
    cursor = conn.cursor()

    query = """
        INSERT INTO prediction_logs (
            applicant_id, is_creditworthy, confidence_score
        ) VALUES (%s, %s, %s)
    """
    values = (applicant_id, is_creditworthy, confidence)

    cursor.execute(query, values)
    conn.commit()
    conn.close()