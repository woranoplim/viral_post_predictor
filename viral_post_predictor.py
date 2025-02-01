import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# โหลดข้อมูล
file_path = "social_media_engagement.csv"
df = pd.read_csv(file_path)

# Normalize metrics to 0-1 range
scaler = MinMaxScaler()
df['normalized_engagement'] = scaler.fit_transform(
    df[['likes', 'shares', 'comments']].sum(axis=1).values.reshape(-1, 1)
)
df['normalized_hour'] = scaler.fit_transform(-df[['post_hour']].values)  # ติดลบเพื่อให้ค่าน้อยมีค่ามาก

# Calculate combined score
df['total_score'] = df['normalized_engagement'] * 0.7 + df['normalized_hour'] * 0.3

# Label top 20%
threshold = df['total_score'].quantile(0.8)
df['viral'] = (df['total_score'] >= threshold).astype(int)

# Train model
features = ["likes", "shares", "comments", "caption_length", "num_hashtags", "post_hour"]
X = df[features]
y = df["viral"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(C=0.01, solver="lbfgs", max_iter=500)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ข้อมูลทดสอบ
sample_data = pd.DataFrame([
    [2500, 1800, 500, 120, 3, 8],  # ตัวอย่างที่ 1
    [3000, 1500, 800, 80, 1, 22],       # ตัวอย่างที่ 2
], columns=["likes", "shares", "comments", "caption_length", "num_hashtags", "post_hour"])

# ทำนาย
predictions = model.predict(sample_data)

# แสดงผลลัพธ์
print(["ไวรัล 🔥" if p == 1 else "ไม่ไวรัล ❄️" for p in predictions])