import mysql.connector

def connect_db():
    conn = mysql.connector.connect(
        host="localhost",           
        user="******",        # encrypted
        password="*******",   # encrypted
        database="CreditWiseAI"    
    )       
    return conn
