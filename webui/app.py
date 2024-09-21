# import streamlit as st

# # Function to check login credentials
# def check_login(username, password):
#     return username == "admin" and password == "admin"

# # Callback functions to update session state and move to the next step
# def go_next_front():
#     st.session_state.front_view = st.session_state["front_view_input"]
#     st.session_state.step = 2
#     if st.session_state.recapture:
#         st.session_state.images_ready = True
#         st.session_state.capturing = False

# def go_next_left():
#     st.session_state.left_view = st.session_state["left_view_input"]
#     st.session_state.step = 3
#     if st.session_state.recapture:
#         st.session_state.images_ready = True
#         st.session_state.capturing = False

# def go_next_right():
#     st.session_state.right_view = st.session_state["right_view_input"]
#     st.session_state.step = 4
#     st.session_state.images_ready = True

# # Function to capture images in sequence
# def capture_images():
#     if "step" not in st.session_state:
#         st.session_state.step = 1

#     if st.session_state.step == 1:
#         st.write("Please capture the front view:")
#         st.camera_input("Capture Front View", key="front_view_input", on_change=go_next_front)

#     elif st.session_state.step == 2:
#         st.write("Please capture the left view:")
#         st.camera_input("Capture Left View", key="left_view_input", on_change=go_next_left)

#     elif st.session_state.step == 3:
#         st.write("Please capture the right view:")
#         st.camera_input("Capture Right View", key="right_view_input", on_change=go_next_right)

# # Initialize session state variables
# if "logged_in" not in st.session_state:
#     st.session_state.logged_in = False
# if "person_name" not in st.session_state:
#     st.session_state.person_name = ""
# if "capturing" not in st.session_state:
#     st.session_state.capturing = False
# if "images_ready" not in st.session_state:
#     st.session_state.images_ready = False
# if "step" not in st.session_state:
#     st.session_state.step = 1
# if "front_view" not in st.session_state:
#     st.session_state.front_view = None
# if "left_view" not in st.session_state:
#     st.session_state.left_view = None
# if "right_view" not in st.session_state:
#     st.session_state.right_view = None
# if "recapture" not in st.session_state:
#     st.session_state.recapture = None

# # Main function
# def main():
#     st.title("Person Image Capture")

#     # Check if user is logged in
#     if not st.session_state.logged_in:
#         # Login form
#         st.subheader("Login")
#         username = st.text_input("Username")
#         password = st.text_input("Password", type="password")
#         if st.button("Login"):
#             if check_login(username, password):
#                 st.session_state.logged_in = True
#                 st.success("Logged in successfully!")
#             else:
#                 st.error("Invalid username or password")
#     else:
#         # Form to enter person's name
#         st.subheader("Enter Person's Name")
#         person_name = st.text_input("Person's Name", value=st.session_state.person_name)
#         if st.button("Start Capturing"):
#             if person_name:
#                 st.session_state.person_name = person_name
#                 st.session_state.capturing = True
#                 st.session_state.images_ready = False
#                 st.session_state.step = 1
#             else:
#                 st.warning("Please enter the person's name")

#         if st.session_state.capturing:
#             capture_images()

#         if st.session_state.images_ready:
#             st.subheader("Preview Captured Images")
#             st.image(st.session_state.front_view, caption="Front View")
#             if st.button("Change Front View"):
#                 st.session_state.step = 1
#                 st.session_state.capturing = True
#                 st.session_state.images_ready = False

#             st.image(st.session_state.left_view, caption="Left View")
#             if st.button("Change Left View"):
#                 st.session_state.step = 2
#                 st.session_state.capturing = True
#                 st.session_state.images_ready = False

#             st.image(st.session_state.right_view, caption="Right View")
#             if st.button("Change Right View"):
#                 st.session_state.step = 3
#                 st.session_state.capturing = True
#                 st.session_state.images_ready = False
            
#             if st.button("Submit"):
#                 # Make API call here
#                 st.success("Images submitted successfully!")
#                 # Reset the state for the next capture
#                 st.session_state.capturing = False
#                 st.session_state.images_ready = False

# if __name__ == "__main__":
#     main()

import streamlit as st

# Function to check login credentials
def check_login(username, password):
    return username == "admin" and password == "admin"

# Callback functions to update session state and move to the next step
def go_next_front():
    st.session_state.front_view = st.session_state["front_view_input"]
    if st.session_state.recapture:
        st.session_state.images_ready = True
        st.session_state.capturing = False
        st.session_state.recapture = False
    else:
        st.session_state.step = 2

def go_next_left():
    st.session_state.left_view = st.session_state["left_view_input"]
    if st.session_state.recapture:
        st.session_state.images_ready = True
        st.session_state.capturing = False
        st.session_state.recapture = False
    else:
        st.session_state.step = 3

def go_next_right():
    st.session_state.right_view = st.session_state["right_view_input"]
    if st.session_state.recapture:
        st.session_state.images_ready = True
        st.session_state.capturing = False
        st.session_state.recapture = False
    else:
        st.session_state.step = 4
        st.session_state.images_ready = True

# Function to capture images in sequence
def capture_images():
    if "step" not in st.session_state:
        st.session_state.step = 1

    if st.session_state.step == 1:
        st.write("Please capture the front view:")
        st.camera_input("Capture Front View", key="front_view_input", on_change=go_next_front)

    elif st.session_state.step == 2:
        st.write("Please capture the left view:")
        st.camera_input("Capture Left View", key="left_view_input", on_change=go_next_left)

    elif st.session_state.step == 3:
        st.write("Please capture the right view:")
        st.camera_input("Capture Right View", key="right_view_input", on_change=go_next_right)

# Initialize session state variables
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "person_name" not in st.session_state:
    st.session_state.person_name = ""
if "capturing" not in st.session_state:
    st.session_state.capturing = False
if "images_ready" not in st.session_state:
    st.session_state.images_ready = False
if "step" not in st.session_state:
    st.session_state.step = 1
if "front_view" not in st.session_state:
    st.session_state.front_view = None
if "left_view" not in st.session_state:
    st.session_state.left_view = None
if "right_view" not in st.session_state:
    st.session_state.right_view = None
if "recapture" not in st.session_state:
    st.session_state.recapture = False

# Main function
def main():
    st.title("Person Image Capture")

    # Check if user is logged in
    if not st.session_state.logged_in:
        # Login form
        st.subheader("Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if check_login(username, password):
                st.session_state.logged_in = True
                st.success("Logged in successfully!")
            else:
                st.error("Invalid username or password")
    else:
        # Form to enter person's name
        st.subheader("Enter Person's Name")
        person_name = st.text_input("Person's Name", value=st.session_state.person_name)
        if st.button("Start Capturing"):
            if person_name:
                st.session_state.person_name = person_name
                st.session_state.capturing = True
                st.session_state.images_ready = False
                st.session_state.step = 1
            else:
                st.warning("Please enter the person's name")

        if st.session_state.capturing:
            capture_images()

        if st.session_state.images_ready:
            st.subheader("Preview Captured Images")
            st.image(st.session_state.front_view, caption="Front View")
            if st.button("Change Front View"):
                st.session_state.step = 1
                st.session_state.capturing = True
                st.session_state.images_ready = False
                st.session_state.recapture = True

            st.image(st.session_state.left_view, caption="Left View")
            if st.button("Change Left View"):
                st.session_state.step = 2
                st.session_state.capturing = True
                st.session_state.images_ready = False
                st.session_state.recapture = True

            st.image(st.session_state.right_view, caption="Right View")
            if st.button("Change Right View"):
                st.session_state.step = 3
                st.session_state.capturing = True
                st.session_state.images_ready = False
                st.session_state.recapture = True
            
            if st.button("Submit"):
                # Make API call here
                st.success("Images submitted successfully!")
                # Reset the state for the next capture
                st.session_state.capturing = False
                st.session_state.images_ready = False

if __name__ == "__main__":
    main()