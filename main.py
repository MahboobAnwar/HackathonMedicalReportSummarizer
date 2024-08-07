import gradio as gr
from pdfminer.high_level import extract_text
from ai71 import AI71
import os

AI71_API_KEY = "ai71-api-170ace63-2fad-4f62-82e1-2d35628dd980"
client = AI71(AI71_API_KEY)

def extract_text_from_pdf(file):
    text = extract_text(file.name)
    return text

def summarize_text(text):
    try:
        response = client.chat.completions.create(
            model="tiiuae/falcon-180B-chat",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"""
I will provide you with a medical report. Please summarize it, including the following information:

1. **Patient Information:**
   - Name
   - Age/Sex
   - Member ID
   - Visit No
   - MR No
   - EID
   - Passport
   - Referred By

2. **Report Type:**
   - Specify if the report is a hematology test report or a diagnostic imaging report.

3. **Hematology Test Results (if applicable):**
   - Sample Date
   - Report Date
   - Detailed results with the following format:
     - Test Name: Result (Units) [Reference Range]
     - Comment: Include a brief comment on whether the result is within the normal range or if there are any abnormalities. If the result is abnormal, provide a possible explanation or implication.

4. **Diagnostic Imaging Results (if applicable):**
   - Study Reason
   - Findings: Provide detailed findings and include any measurements or observations.
   - Impression: Summarize the overall impression of the findings.
   - Recommendations: Provide any recommendations for further evaluation or follow-up.

5. **Areas for Improvement:**
   - Identify any test results or findings that are outside the normal range and provide recommendations for improvement or further investigation.
   - Include possible medical conditions or concerns that should be addressed based on the abnormal results or findings.

6. **Suggested Diet:**
   - Based on the identified areas for improvement, suggest a comprehensive diet plan to address these issues. Include foods rich in specific nutrients that can help improve the identified areas of concern.

7. **Follow-Up Recommendations:**
   - Provide detailed follow-up recommendations based on the test results or findings. Include advice on how frequently the patient should monitor their health, any additional tests that might be required, and any lifestyle changes that could help improve their health.

Here is the medical report:
{text}
                """},
            ],
        )
        print(response)  # Log the response
        summary = response.choices[0].message.content  # Adjusted to access the content correctly
        return summary
    except Exception as e:
        print(f"Error: {e}")  # Log the error
        return f"An error occurred: {e}"

def chat_with_falcon(chat_history, user_message, extracted_text):
    if chat_history is None:
        chat_history = []

    chat_history.append({"role": "user", "content": user_message})

    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    messages += [{"role": msg["role"], "content": msg["content"]} for msg in chat_history]
    messages.append({"role": "user", "content": f"Here is the medical report: {extracted_text}"})

    response = client.chat.completions.create(
        model="tiiuae/falcon-180B-chat",
        messages=messages,
    )
    falcon_reply = response.choices[0].message.content
    chat_history.append({"role": "assistant", "content": falcon_reply})

    # Format chat history for Gradio's Chatbot component
    formatted_history = [(msg["role"], msg["content"]) for msg in chat_history]

    return chat_history, formatted_history

def process_pdf(file):
    extracted_text = extract_text_from_pdf(file)
    summary = summarize_text(extracted_text)
    return extracted_text, summary

# Create the Gradio interface
upload_interface = gr.Interface(
    fn=process_pdf,
    inputs=gr.File(label="Upload PDF"),
    outputs=[
        gr.Textbox(label="Extracted Text"),
        gr.Textbox(label="Summary"),
    ],
    allow_flagging='never',
    title="Medical Report Summarizer with Falcon 2",
    description="Upload a Medical Report in PDF form",
)

chat_interface = gr.Interface(
    fn=chat_with_falcon,
    inputs=[
        gr.State(),  # chat history
        gr.Textbox(label="Your Message"),
        gr.Textbox(label="Extracted Text")  # Add extracted text as input
    ],
    allow_flagging='never',
    outputs=[
        gr.State(),  # updated chat history
        gr.Chatbot(label="Chat with Falcon")
    ],
    title="Chat with Falcon",
    description="Ask questions about your medical report or medical terminology."
)

# Combine the interfaces into a tabbed interface
iface = gr.TabbedInterface([upload_interface, chat_interface], ["Upload PDF", "Chat with Falcon"])

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))  # Use the PORT environment variable set by Render
    iface.launch(server_name="0.0.0.0", server_port=port)
