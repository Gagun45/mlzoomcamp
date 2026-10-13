import pickle

input_file = './pipeline.bin'

record = {
  "lead_source": "paid_ads",
  "industry": "technology",
  "employment_status": "employed",
  "location": "north_america",
  "number_of_courses_viewed": 2,
  "annual_income": 79276.0,
  "interaction_count": 4,
  "lead_score": 0.41
}

with open(input_file, 'rb') as f_in:
    pipeline = pickle.load(f_in)
probability = float(pipeline.predict_proba([record])[0, 1])
print("conversion_probability: ", round(probability, 3))
