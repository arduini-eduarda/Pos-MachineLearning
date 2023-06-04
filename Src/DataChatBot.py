import os
import sqlite3

def create_connection():
    conn = None
    try:
        path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        db = os.path.join(path, 'Repository\menu.db')
        conn = sqlite3.connect(db)
    except:
        print("Connection error")

    return conn

def getCat():
    listCat = []
    cnx = create_connection()
    with cnx:
        cur = cnx.cursor()
        cur.execute("SELECT Name FROM Category")

        rows = cur.fetchall()

        for row in rows:
            listCat.append(row)

    return listCat

def getCat():
    listCat = []
    cnx = create_connection()
    with cnx:
        cur = cnx.cursor()
        cur.execute("SELECT Name FROM Category")

        rows = cur.fetchall()

        for row in rows:
            listCat.append(row)

    return listCat


def createData():

    listCat = getCat()
    categories = ""

    for l in listCat:
        categories = categories + " , " + l[0]

    jsonChatBot = ""

    f = open("Data/intents.json", "r")
    jsonChatBot = f.read()
    f.close()

    text = jsonChatBot.replace("{categories}", categories)

    new_file = open("Data/intents_new.json", "a")
    new_file.write(text)
    new_file.close()

