
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  LabelEncoder
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import matplotlib.dates as mdates
import seaborn as sns


# Set page config
st.set_page_config(
    page_title="Netflix Stock Price Prediction App",
    page_icon=":movie_camera:",
    #layout="wide",
    initial_sidebar_state="collapsed",
)

menu = ["Home","Services", "About", "Contact"]
choice = st.sidebar.selectbox("Select Page", menu)


if choice == "Home":
    
     # Add your existing home page code here
    sns.set_style('darkgrid')

    # Define CSS styles for text elements
    HEADER_STYLE = """
        font-size: 50px;
        font-family: 'Arial Black', sans-serif;
        font-weight: bold;
        color: #FF9633;
        text-align: center;
        margin-top: 50px;
        margin-bottom: 50px;
    """

    TEXT_STYLE = """
        font-size: 20px;
        font-family: 'Arial', sans-serif;
        color: #2F4F4F;
        text-align: center;
    """



    # Set page title and header

    st.markdown("<p style='" + HEADER_STYLE + "'>Netflix Stock Price Prediction App</p>", unsafe_allow_html=True)
    st.write("This Web app can be used for predicting Netflix stock prices for a specified number of days using the historical data. The visualization of the historical data regarding the inflation/decrease in rates of stocks with their time series components can also be observed.")
 

    


    # Define custom CSS style with background image
    import base64
    def add_bg_from_local(image_file):
        with open(image_file, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
        st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
            background-size: cover
        }}
        </style>
        """,
        unsafe_allow_html=True
        )

    add_bg_from_local('stock_price1.jpg')




elif choice == "Services":
    # Add your existing home page code here
    sns.set_style('darkgrid')

    # Define CSS styles for text elements
    HEADER_STYLE = """
        font-size: 50px;
        font-family: 'Arial Black', sans-serif;
        font-weight: bold;
        color: #FF9633;
        text-align: center;
        margin-top: 50px;
        margin-bottom: 50px;
    """

    TEXT_STYLE = """
        font-size: 20px;
        font-family: 'Arial', sans-serif;
        color: #2F4F4F;
        text-align: center;
    """



    # Set page title and header
    st.markdown("<p style='" + HEADER_STYLE + "'>Netflix Stock Price Prediction App</p>", unsafe_allow_html=True)


    # Define custom CSS style with background image
    import base64
    def add_bg_from_local(image_file):
        with open(image_file, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
        st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
            background-size: cover
        }}
        </style>
        """,
        unsafe_allow_html=True
        )

    add_bg_from_local('stock_price1.jpg')    

    # Define custom CSS style with fonts
    st.markdown(""" 
        <style> 
            .font {
                font-size:50px ; 
                font-family: 'Red'; 
                color: #FF9633;
            }
            .text {
                font-size:20px ;
                font-family: 'Red'; 
                color: #000000;
            }
            .stButton button {
                background-color: #FF9633;
                color: #FFFFFF;
                border-radius: 8px;
                border: none;
                font-weight: bold;
                font-size: 16px;
            }
        </style> 
    """, unsafe_allow_html=True)


    SLIDER_STYLE = """
    <style>
    [data-testid="stSlider"] .streamlit-slider { 
        background-color: #f63366;
    }
    </style>
    """

    st.markdown(SLIDER_STYLE, unsafe_allow_html=True)

    with st.container():
        st.write("### Days")
        with st.container():
            input_days = st.slider("Select a value", 0, 60, 30)

    # Load the data
    df = pd.read_csv('NFLX.csv', index_col='Date', parse_dates=True)



    def predict_stocks(number):
        df = pd.read_csv('NFLX.csv', index_col='Date', parse_dates=True)

        model = tf.keras.models.load_model('best_lstm_model.h5', compile = False)
        model.compile(optimizer='adam', loss='mse')
        # Sort the data by date
        df = df.sort_values('Date')

        # Create a new dataframe with only the 'Close' column
        data = df.filter(['Adj Close'])

        # Convert the dataframe to a numpy array
        dataset = data.values

        # Scale the data
        scaler = MinMaxScaler(feature_range=(0,1))
        scaled_data = scaler.fit_transform(dataset)

        # Define the number of days to predict in the future
        prediction_days = number

        # Create a list of dates for the prediction period
        last_date = df.index[-1]
        dates = pd.date_range(last_date, periods=prediction_days+1, freq='B')[1:]

        # Predict the future prices
        last_60_days = scaled_data[-60:]
        X_predict = []
        X_predict.append(last_60_days)
        X_predict = np.array(X_predict)
        X_predict = np.reshape(X_predict, (X_predict.shape[0], X_predict.shape[1], 1))

        predicted_prices = []

        for i in range(prediction_days):
            predicted_price = model.predict(X_predict,verbose=0)
            predicted_prices.append(predicted_price[0])
            last_60_days = np.append(last_60_days[1:], predicted_price, axis=0)
            X_predict = np.array([last_60_days])
            X_predict = np.reshape(X_predict, (X_predict.shape[0], X_predict.shape[1], 1))

        # Inverse transform the predicted prices to their original scale
        predicted_prices = scaler.inverse_transform(predicted_prices)

        # Create a dataframe of the predicted prices and dates
        predictions = pd.DataFrame(predicted_prices, index=dates, columns=['Adj Close'])

        # Define colors
        actual_color = '#00BFFF'  # blue
        predicted_color = '#FFA500'  # orange
        background_color = '#F5F5F5'  # light gray
        title_color = '#1E90FF'  # dodger blue

        # Plot the original and predicted stock prices
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.plot(data['Adj Close'], color=actual_color, label='Actual')
        ax.plot(predictions['Adj Close'], color=predicted_color, label='Predicted')
        ax.set_title('Netflix Stock Price Prediction', fontsize=28, color=title_color, fontweight='bold')
        ax.set_xlabel('Date', fontsize=18, fontweight='bold')
        ax.set_ylabel('Close Price', fontsize=18, fontweight='bold')
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.legend(fontsize=18)

        # Set the background color
        fig.patch.set_facecolor(background_color)
        ax.set_facecolor(background_color)

        # Format x-axis ticks as dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

        # Display the plot in Streamlit app
        st.pyplot(fig)

        return predictions


    # To display the current trend of the netflix stock prices
    if st.checkbox('Display current value of stock prices'):
        st.dataframe(df['Adj Close'].tail(5))

    # To predict the future stock prices according to the input given
    if st.button('Make Prediction'):
        prediction = predict_stocks(input_days)
        # Define colors
        background_color = '#F5F5F5'  # light gray
        header_color = '#1E90FF'  # dodger blue
        cell_color = '#D3D3D3'  # light gray

        # Style the DataFrame
        styled_df = prediction.style \
            .set_properties(**{'background-color': background_color, 'color': 'black'}) \
            .set_table_styles([{'selector': 'th', 'props': [('background-color', header_color), ('color', 'white')]}]) \
            .highlight_max(color=cell_color) \
            .highlight_min(color=cell_color)
        st.write("Predicted Value", styled_df)

elif choice == "About":
    st.markdown("""
    <style>
        h1 {
            color: #336699;
            font-size: 36px;
            font-weight: bold;
            text-align: center;
            margin-bottom: 30px;
        }
        h2 {
            color: #336699;
            font-size: 24px;
            font-weight: bold;
            margin-bottom: 20px;
        }
        p {
            font-size: 18px;
            margin-bottom: 20px;
        }
    </style>
    """, unsafe_allow_html=True)

    st.write("<h1>About this app</h1>", unsafe_allow_html=True)
    st.write("<p>The app contains options to visualize the stock price trends and also view the stock price for the number of days provided. The user has to input the number of days for which the stock price can be predicted.</p>", unsafe_allow_html=True)
    st.write("<p>If user enters the desired number of days for Netflix stock price, the web app displays the visualization which gives the increase or decrease in trends in Netflix stock prices for the particular number of days entered by the user. The user also can see the latest data on Netflix stock prices to make a decision on whether or not the seasonality and trend continue in the data.</p>", unsafe_allow_html=True)
    st.write("<p>Our models can then make predictions about future stock prices, based on the input data you provide.</p>", unsafe_allow_html=True)
    st.write("<br><h2>About the developers</h2>", unsafe_allow_html=True)
    st.write("<p>This app is developed by  Shruthi Senthilmani, Akash Patil and Shiva Reddy.</p>", unsafe_allow_html=True)
    st.write("<p>If you have any feedback or questions about this app, please don't hesitate to get in touch with us!</p>", unsafe_allow_html=True)


elif choice== "Contact":
    st.markdown("""
    <style>
        h1 {
            color: #336699;
            font-size: 36px;
            font-weight: bold;
            text-align: center;
            margin-bottom: 30px;
        }
        h2 {
            color: #336699;
            font-size: 24px;
            font-weight: bold;
            margin-bottom: 20px;
        }
        p {
            font-size: 18px;
            margin-bottom: 20px;
        }
    </style>
    """, unsafe_allow_html=True)


    st.write("<h1>Contact Us</h1>", unsafe_allow_html=True)
    st.write("<p>This app is developed by Shruthi Senthilmani, Akash Patil and Shiva Reddy.</p>", unsafe_allow_html=True)
    st.write("<p> We are Data Science graduate students from Indiana University Bloomington.</p>", unsafe_allow_html=True)
    st.write("<p>If you have any feedback or questions about this app, please don't hesitate to get in touch with us!</p>", unsafe_allow_html=True)

    st.markdown("""<form action="//submit.form" id="ContactUs100" method="post" onsubmit="return ValidateForm(this);">
<script type="text/javascript">
function ValidateForm(frm) {
if (frm.Name.value == "") { alert('Name is required.'); frm.Name.focus(); return false; }
if (frm.FromEmailAddress.value == "") { alert('Email address is required.'); frm.FromEmailAddress.focus(); return false; }
if (frm.FromEmailAddress.value.indexOf("@") < 1 || frm.FromEmailAddress.value.indexOf(".") < 1) { alert('Please enter a valid email address.'); frm.FromEmailAddress.focus(); return false; }
if (frm.Comments.value == "") { alert('Please enter comments or questions.'); frm.Comments.focus(); return false; }
return true; }
</script>
<table style="width:100%;max-width:550px;border:0;" cellpadding="8" cellspacing="0">
<tr> <td>
<label for="Name">Name*:</label>
</td> <td>
<input name="Name" type="text" maxlength="60" style="width:100%;max-width:250px;" />
</td> </tr> <tr> <td>
<label for="PhoneNumber">Phone number:</label>
</td> <td>
<input name="PhoneNumber" type="text" maxlength="43" style="width:100%;max-width:250px;" />
</td> </tr> <tr> <td>
<label for="FromEmailAddress">Email address*:</label>
</td> <td>
<input name="FromEmailAddress" type="text" maxlength="90" style="width:100%;max-width:250px;" />
</td> </tr> <tr> <td>
<label for="Comments">Comments*:</label>
</td> <td>
<textarea name="Comments" rows="7" cols="40" style="width:100%;max-width:350px;"></textarea>
</td> </tr> <tr> <td>
* - required fields
</td> <td>

<input name="skip_Submit" type="submit" value="Submit" />
<script src="https://www.100forms.com/js/FORMKEY:JJ2X2JH45GW9/SEND:my@email.com" type="text/javascript"></script>
</td> </tr>
</table>
</form>""",unsafe_allow_html=True)


