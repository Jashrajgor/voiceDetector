import base64

with open("harvard.wav", "rb") as f:
    encoded = base64.b64encode(f.read()).decode()

with open("audio_base64.txt", "w") as f:
    f.write(encoded)

print("encoded successfully")