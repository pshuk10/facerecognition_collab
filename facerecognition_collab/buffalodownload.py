import insightface

model = insightface.app.FaceAnalysis(name="buffalo_l")
model.prepare(ctx_id=0)
print("Model files downloaded to:", model.model_store)
