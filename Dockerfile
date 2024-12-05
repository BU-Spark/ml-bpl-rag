FROM python:3.9







# Create a non-root user



RUN useradd -m -u 1000 user



USER user







# Set PATH to include user's local bin



ENV PATH="/home/user/.local/bin:$PATH"







# Set working directory



WORKDIR /app







# Copy requirements file with appropriate ownership



COPY --chown=user ./requirements.txt requirements.txt







# Install dependencies



RUN pip install --no-cache-dir -r requirements.txt







# Copy application files with appropriate ownership



COPY --chown=user . /app







# Set environment variables for Streamlit



ENV HOST=0.0.0.0



ENV PORT=7860



ENV STREAMLIT_SERVER_PORT=7860




ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0






# Change the CMD to use chainlit

CMD ["streamlit", "run", "streamlit-rag-app.py", "--server.port", "7860", "--server.address", "0.0.0.0"]