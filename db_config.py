import mysql.connector

def connect_db():
    conn = mysql.connector.connect(
        host="localhost",           
        user="******",            # schema user name /ex: root etc. 
        password="******",        # schema password
        database="CreditWiseAI"    
    )       
    return conn
