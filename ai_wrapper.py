import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def generate_explanation(prediction: str, data) -> str:
    prompt = f"""
        Anda adalah Dr. PilHeart, seorang dokter spesialis jantung dengan pengalaman bertahun-tahun.
        Anda sedang memberikan konsultasi kepada pasien yang baru saja menerima hasil prediksi penyakit jantung.

        Hasil prediksi: {prediction}
        Data pasien: {data}

        Tugas Anda:
        1. Jelaskan secara jelas dan sederhana apa arti hasil prediksi ini agar pasien awam dapat memahami.
        2. Berikan saran medis yang bermanfaat dan empatik berdasarkan hasil prediksi serta data pasien.
        3. Sertakan rekomendasi gaya hidup seperti pola makan, olahraga, dan pengelolaan stres yang sesuai.
        4. Gunakan nada profesional, menenangkan, dan penuh perhatian — seperti dokter yang sedang berbicara langsung kepada pasiennya.
        5. Jika hasil menunjukkan adanya risiko, jelaskan dengan hati-hati tanpa menimbulkan panik, serta arahkan langkah medis yang sebaiknya dilakukan selanjutnya.
        6. Pastikan jawaban Anda lengkap hingga akhir, jangan dipotong.
        7. Berikan penjelasan singkat saja 2 paragraf
        8. Pastikan Anda Memberi Salam Kepada Pasien
        9. Tutup penjelasan dengan kalimat penutup yang menenangkan, misalnya “Semoga Anda tetap sehat dan tenang.”

    """

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful AI medical assistant.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_tokens=1500,
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"Failed to generate explanation: {str(e)}"
