import streamlit as st
import sys

st.set_option('client.showErrorDetails', True)

def main():
    try:
        st.title("Test App")
        st.write("Hello World!")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        print(f"Error: {str(e)}", file=sys.stderr)

if __name__ == "__main__":
    main()