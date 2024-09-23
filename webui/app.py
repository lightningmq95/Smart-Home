import streamlit as st
import requests

def go_next(view):
    if view == "front":
        st.session_state.front_view = st.session_state["front_view_input"]
        next_step = 2
    elif view == "left":
        st.session_state.left_view = st.session_state["left_view_input"]
        next_step = 3
    elif view == "right":
        st.session_state.right_view = st.session_state["right_view_input"]
        next_step = 4
        st.session_state.images_ready = True

    if st.session_state.recapture:
        st.session_state.images_ready = True
        st.session_state.capturing = False
        st.session_state.recapture = False
    else:
        st.session_state.step = next_step

def change_view(view):
    if view == "front":
        st.session_state.step = 1
    elif view == "left":
        st.session_state.step = 2
    elif view == "right":
        st.session_state.step = 3
    st.session_state.capturing = True
    st.session_state.images_ready = False
    st.session_state.recapture = True

def submit_images():
    # Make API call here
    files = [
        ('images', ('front_view.jpg', st.session_state.front_view.getvalue(), 'image/jpeg')),
        ('images', ('left_view.jpg', st.session_state.left_view.getvalue(), 'image/jpeg')),
        ('images', ('right_view.jpg', st.session_state.right_view.getvalue(), 'image/jpeg'))
    ]
    data = {
        'name': st.session_state.person_name,
        'role': st.session_state.role  
    }
    response = requests.post('http://127.0.0.1:8000/register_user', files=files, data=data)
    if response.status_code == 200:
        st.success("Images submitted successfully!")
    else:
        st.error("Failed to submit images.")
    # Reset the state for the next capture
    st.session_state.capturing = False
    st.session_state.images_ready = False

def start_capturing(person_name, role):
    if person_name:
        st.session_state.person_name = person_name
        st.session_state.role = role
        st.session_state.capturing = True
        st.session_state.images_ready = False
        st.session_state.step = 1
    else:
        st.warning("Please enter the person's name")

def change_password(old_password, new_password):
    response = requests.post('http://127.0.0.1:8000/change_password', data={'username': st.session_state.username, 'old_password': old_password, 'new_password': new_password})
    if response.status_code == 200:
        st.success("Password changed successfully!")
    else:
        st.error("Failed to change password. Please check your old password.")

@st.dialog("Change Password")
def show_change_password_modal():
    old_password = st.text_input("Old Password", type="password")
    new_password = st.text_input("New Password", type="password")
    st.button("Submit", on_click=change_password, args=(old_password, new_password))


# Initialize session state variables
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "person_name" not in st.session_state:
    st.session_state.person_name = ""
if "role" not in st.session_state:
    st.session_state.role = "owner"
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

role_list = ['owner', 'member']

def login(username, password):
    response = requests.post('http://127.0.0.1:8000/login', data={'username': username, 'password': password})
    if response.status_code == 200:
        st.session_state.logged_in = True
        st.session_state.username = username
        st.success("Logged in successfully!")
    else:
        st.error("Invalid username or password")

# Main function
def main():
    st.title("Person Image Capture")

    # Check if user is logged in
    if not st.session_state.logged_in:
        # Login form
        st.subheader("Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        st.button("Login", on_click=login, args=(username, password))

    else:
        st.button("Change Password", on_click=show_change_password_modal)

        # Form to enter person's name
        st.subheader("Enter Person's Name")
        person_name = st.text_input("Person's Name", value=st.session_state.person_name)
        role = st.selectbox("Select Role", role_list, index=role_list.index(st.session_state.role))

        st.button("Start Capturing", on_click=start_capturing, args=(person_name, role))

        if st.session_state.capturing:
            if "step" not in st.session_state:
                st.session_state.step = 1

            if st.session_state.step == 1:
                st.write("Please capture the front view:")
                st.camera_input("Capture Front View", key="front_view_input", on_change=go_next, args=("front",))

            elif st.session_state.step == 2:
                st.write("Please capture the left view:")
                st.camera_input("Capture Left View", key="left_view_input", on_change=go_next, args=("left",))

            elif st.session_state.step == 3:
                st.write("Please capture the right view:")
                st.camera_input("Capture Right View", key="right_view_input", on_change=go_next, args=("right",))

        if st.session_state.images_ready:
            st.subheader("Preview Captured Images")

            st.image(st.session_state.front_view, caption="Front View")
            st.button("Change Front View", on_click=change_view, args=("front",))

            st.image(st.session_state.left_view, caption="Left View")
            st.button("Change Left View", on_click=change_view, args=("left",))

            st.image(st.session_state.right_view, caption="Right View")
            st.button("Change Right View", on_click=change_view, args=("right",))
            
            st.button("Submit", on_click=submit_images)

if __name__ == "__main__":
    main()