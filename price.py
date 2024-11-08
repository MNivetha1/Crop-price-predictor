import pandas as pd
from flask import Flask, request, jsonify, render_template
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
try:
    df = pd.read_csv('top_20_commodities.csv')  
    print("Dataset loaded successfully.")
    print("Columns in dataset:", df.columns)
except FileNotFoundError:
    print("Error: The file 'top_20_commodities.csv' was not found.")
    exit()
required_columns = [
    'State', 'District', 'Commodity', 'Variety', 'Grade', 
    'Min_Price', 'Max_Price', 'Modal_Price', 'Commodity_Code']
df = df.dropna(subset=['Min_Price', 'Max_Price', 'Modal_Price'])
df = pd.get_dummies(df, columns=['Market'], drop_first=True)
features = ['Commodity_Code', 'Min_Price', 'Max_Price'] + [col for col in df.columns if 'Market_' in col]
X = df[features]
y = df['Modal_Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
decision_regressor = DecisionTreeRegressor(random_state=42)
decision_regressor.fit(X_train, y_train)
app = Flask(__name__)
commodities = df['Commodity'].unique().tolist()
encoded_columns = [col for col in df.columns if 'Market_' in col]
markets = [col.replace('Market_', '') for col in encoded_columns]
@app.route('/')
def index():
    selected_commodity = request.args.get('commodity', None)
    if selected_commodity:
        filtered_df = df[df['Commodity'] == selected_commodity]
        data = filtered_df[required_columns + encoded_columns].to_dict(orient='records')
    else:
        data = []
    return render_template('price.html', commodities=commodities, markets=markets, data=data, selected_commodity=selected_commodity)
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    commodity = data.get('commodity')
    market = data.get('market')
    filtered_df = df[df['Commodity'] == commodity]
    if filtered_df.empty:
        return jsonify({'error': 'No data found for the selected commodity'}), 404
    latest_entry = filtered_df.iloc[-1]
    market_col = f'Market_{market}'
    input_data = pd.DataFrame([[latest_entry['Commodity_Code'], 
                                latest_entry['Min_Price'], latest_entry['Max_Price']]], 
                               columns=['Commodity_Code', 'Min_Price', 'Max_Price'])
    for col in encoded_columns:
        input_data[col] = 1 if col == market_col else 0
    predicted_price = decision_regressor.predict(input_data)[0]
    return jsonify({
        'predicted_price': round(predicted_price, 2),
        'commodity': commodity,
        'market': market,
        'filtered_data': filtered_df[required_columns].to_dict(orient='records')
    })
if __name__ == '__main__':
    app.run(debug=True)
