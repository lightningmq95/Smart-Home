from click import option
import streamlit as st
# from streamlit_navigation_bar import st_navbar
from streamlit_option_menu import option_menu

import requests
import pages as pg

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
if "gemini_response" not in st.session_state:
    st.session_state.gemini_response = ""

role_list = ['owner', 'member']

def login(username, password):
    response = requests.post('http://127.0.0.1:8000/login', data={'username': username, 'password': password})
    if response.status_code == 200:
        st.session_state.logged_in = True
        st.session_state.username = username
        st.success("Logged in successfully!")
    else:
        st.error("Invalid username or password")

pages = ["Add Person", "Login", "Integrated Page"]
functions = {
    "Add Person": lambda: pg.add_user(on_click=login),
    "Login": lambda: pg.login(on_click=login),
    "Integrated Page": pg.integrated_page,

}

# Main function
def main():
    # page = st_navbar(pages)
    page = option_menu(
        menu_title=None,
        options=pages,
        icons=["graph-up-arrow", "basket", "graph-down"],
        default_index=0,
        orientation="horizontal",
    )
    # Check if user is logged in
    go_to = functions.get(page)
    if go_to:
        go_to()
if __name__ == "__main__":
    main()