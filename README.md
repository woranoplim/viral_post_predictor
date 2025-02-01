# Viral Post Predictor 🚀

โมเดล Machine Learning สำหรับทำนายว่าโพสต์ในโซเชียลมีเดียจะเป็นไวรัลหรือไม่ โดยอิงจากการมีส่วนร่วมของผู้ใช้ เช่น จำนวนไลก์ แชร์ คอมเมนต์ จำนวนขั่วโมงที่โพสต์และปัจจัยอื่น ๆ

## 🔍 วิธีการทำงาน
1. โหลดข้อมูลจากไฟล์ `social_media_engagement.csv`
2. ปรับขนาดค่าตัวแปร (Normalization) และคำนวณคะแนนรวมของโพสต์
3. กำหนดเกณฑ์ให้โพสต์ที่มีคะแนนสูงกว่า 20% บนสุดเป็น "ไวรัล"
4. ฝึกโมเดล Logistic Regression เพื่อทำการทำนาย
5. ทดสอบโมเดลและแสดงผลลัพธ์

#accuracy 0.99

## 🤖 จัดทำโดย  
วรนพ ลิมป์ปีติวรกุล และ AI Support (Claude, ChatGPT)

## 📂 แหล่งที่มา  
**Dataset:** [The Power of Social Media Engagement](https://www.kaggle.com/datasets/ashaychoudhary/the-power-of-social-media-engagement)  
