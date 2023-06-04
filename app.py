import pickle
import streamlit as st
import numpy as np
import pandas as pd

# Load the necessary data
model = pickle.load(open('artifacts/model.pkl', 'rb'))
book_names = pickle.load(open('artifacts/book_names.pkl', 'rb'))
final_rating = pickle.load(open('artifacts/final_rating.pkl', 'rb'))
book_pivot = pickle.load(open('artifacts/book_pivot.pkl', 'rb'))

st.markdown(
    """
    <style>
    .sidebar .sidebar-content {
        width: 320px;
    }
    .sidebar .sidebar-list {
        font-size: 20px;
        padding-top: 20px;
        padding-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Apply custom CSS style to increase size of header
st.markdown(
    """
    <style>
    h1 {
        font-size: 36px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Home Page
def display_home():
    st.title('Book Recommender System')
    st.write('Welcome to the Book Recommender System app!')
    st.write('This app utilizes a machine learning model to recommend books based on user preferences.')
    st.write('You can select a book from the dropdown menu and click "Show Recommendation" to get personalized book recommendations.')
    st.write('Additionally, you can navigate to the Top 50 Books page to view the top 50 books based on their ratings.')
    st.write('Enjoy exploring the world of books!')
    # Add any additional content or formatting as desired

# HTML Page
def display_html():
    st.header('Top 50 Books')
    # Load the final_rating DataFrame
    final_rating = pickle.load(open('artifacts/final_rating.pkl', 'rb'))

    # Drop duplicates based on the 'title' column
    final_rating = final_rating.drop_duplicates(subset='title')

    # Load and display the top 50 unique books based on rating
    top_books = final_rating.nlargest(50, 'rating')

    num_cols = 3  # Number of columns to display the books

    # Calculate the number of rows and columns needed
    num_rows = len(top_books) // num_cols
    if len(top_books) % num_cols != 0:
        num_rows += 1

    # Divide the books into groups of 3 and display them in separate columns
    cols = [st.columns(num_cols) for _ in range(num_rows)]
    for i, col_group in enumerate(cols):
        for j, col in enumerate(col_group):
            index = i * num_cols + j
            if index < len(top_books):
                with col:
                    st.text(top_books.iloc[index]['title'])
                    st.image(top_books.iloc[index]['image_url'])


# Book Recommender System Page
def display_book_recommender():
    st.header('Machine Learning Book Recommender System')

    def fetch_poster(suggestion):
        book_name = []
        ids_index = []
        poster_url = []

        for book_id in suggestion:
            book_name.append(book_pivot.index[book_id])

        for name in book_name[0]:
            ids = np.where(final_rating['title'] == name)[0][0]
            ids_index.append(ids)

        for idx in ids_index:
            url = final_rating.iloc[idx]['image_url']
            poster_url.append(url)

        return poster_url

    def recommend_book(book_name):
        books_list = []
        book_id = np.where(book_pivot.index == book_name)[0][0]
        distance, suggestion = model.kneighbors(book_pivot.iloc[book_id, :].values.reshape(1, -1), n_neighbors=6)

        poster_url = fetch_poster(suggestion)

        for i in range(len(suggestion)):
            books = book_pivot.index[suggestion[i]]
            for j in books:
                books_list.append(j)
        return books_list, poster_url

    selected_books = st.selectbox(
        "Type or select a book from the dropdown",
        book_names
    )

    if st.button('Show Recommendation'):
        recommended_books, poster_url = recommend_book(selected_books)
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.text(recommended_books[1])
            st.image(poster_url[1])
        with col2:
            st.text(recommended_books[2])
            st.image(poster_url[2])

        with col3:
            st.text(recommended_books[3])
            st.image(poster_url[3])
        with col4:
            st.text(recommended_books[4])
            st.image(poster_url[4])
        with col5:
            st.text(recommended_books[5])
            st.image(poster_url[5])

# Sidebar Navigation
nav_selection = st.sidebar.radio("Navigation", ("Home", "Book Recommender", "Top 50 Books"))

# Handle Navigation Selection
if nav_selection == "Home":
    display_home()
elif nav_selection == "Book Recommender":
    display_book_recommender()
else:
    display_html()

