from deepface import DeepFace

# Paths to your two images
img1_path = "pranay_14.jpg"
img2_path = "pranay_15.jpg"

# Compare using DeepFace.verify()
result = DeepFace.verify(img1_path, img2_path)

# Print the result
print("✅ Verification Result:")
print(result)

# Show the main outcome clearly
if result["verified"]:
    print("\n✅ The two images belong to the same person!")
else:
    print("\n❌ The two images are of different people.")
