import sqlite3


DB_PATH = "prediction.db"


def add_ai_summary_column():

    conn = sqlite3.connect(DB_PATH)

    cursor = conn.cursor()


    cursor.execute(
        "PRAGMA table_info(predictions)"
    )


    columns = [
        row[1]
        for row in cursor.fetchall()
    ]


    if "ai_summary" not in columns:

        cursor.execute(
            """
            ALTER TABLE predictions
            ADD COLUMN ai_summary TEXT
            """
        )

        print(
            "新增 ai_summary 欄位完成"
        )

    else:

        print(
            "ai_summary 欄位已存在"
        )


    conn.commit()
    conn.close()



if __name__ == "__main__":

    add_ai_summary_column()