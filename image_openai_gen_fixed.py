from openai import OpenAI
client = OpenAI()

response = client.images.edits(
  model="gpt-image-1",
  image=open("seventh.png", "rb"),
  prompt="Black and red ergonomic gaming chair in front of a triple monitor setup, RGB-lit keyboard, ambient room lighting, ideal for gamers. Person seated from behind with hands on keyboard, face not shown.",
  n=1,
  size="1024x1024",
  quality="high",
  style="natural"
)

# Print the URL of the generated image
print(response.data[0].url)
