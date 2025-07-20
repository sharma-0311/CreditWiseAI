import mysql.connector

def connect_db():
    conn = mysql.connector.connect(
        host="localhost",           
        user="credit_user",     
        password="credit_pass",
        database="CreditWiseAI"    
    )       
    return conn
