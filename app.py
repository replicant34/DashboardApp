# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------
# Libraries
# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, dash_table
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from mlxtend.frequent_patterns import apriori, association_rules
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
import networkx as nx
from dash.dependencies import Input, Output, State
from dash.dependencies import State 

# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------
# Data Prep
# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------

# read the file
df = pd.read_csv("file.csv")
df2 = df.copy()
df3 = df.copy()
# feature selection
df = df[['CustomerID',
         'Gender',
         'Location',
         'Transaction_ID',
         'Transaction_Date',
         'Product_Description',
         'Product_Category',
         'Quantity',
         'Avg_Price',
         'Online_Spend',
         'Offline_Spend']]

# convert states names into codes
location_mapping = {
    'Chicago': 'IL',
    'California': 'CA',
    'New York': 'NY', 
    'New Jersey': 'NJ', 
    'Washington DC': 'DC',   
}

df['Location_Code'] = df['Location'].map(location_mapping)

# Get total spent 
df['Total_Spend'] = df['Offline_Spend'] + df['Online_Spend']

# Adjust dates
df['Transaction_Date'] = pd.to_datetime(df['Transaction_Date'])
df['Month'] = pd.to_datetime(df['Transaction_Date']).dt.to_period('M')

# Drop missing values
df = df.dropna()

# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------
# Data Prep for recomendation user-user
# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------

recommendation_df = df2[["CustomerID", "Product_Description"]]
recommendation_df['Rank'] = recommendation_df.groupby(['CustomerID', 'Product_Description'])['Product_Description'].transform('count')
recommendation_df = recommendation_df.drop_duplicates()
recommendation_df = recommendation_df.dropna()
recommendation_df = recommendation_df[recommendation_df.groupby('CustomerID')['CustomerID'].transform('count') > 1]

# User-item matrix
user_item_matrix = recommendation_df.pivot_table(index='CustomerID', columns='Product_Description', values='Rank', fill_value=0)

# Train-test split
X_train, X_test = train_test_split(user_item_matrix, test_size=0.25, random_state=42)

# Train the collaborative filtering model using KNN
model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
model_knn.fit(X_train)

# Define function for recommendation user-user
def user_user(user_id, num_recommendations=1):
    if user_id not in user_item_matrix.index:
        return [f"User ID {user_id} not found in the dataset."]
    
    # get row index for the user
    user_row = user_item_matrix.loc[user_id].values.reshape(1, -1)
    
    # Find similar users
    distances, indices = model_knn.kneighbors(user_row, n_neighbors=num_recommendations + 1)
    similar_user_indices = indices.flatten()[1:]
    
    # Get items from similar users
    similar_users = user_item_matrix.iloc[similar_user_indices]
    recommendations = similar_users.sum(axis=0)
    
    # Exclude products the user already owns
    user_products = set(user_item_matrix.loc[user_id][user_item_matrix.loc[user_id] > 0].index)
    recommendations = recommendations.drop(user_products)
    
    # Get top recommendations
    recommended_products = recommendations.nlargest(num_recommendations).index.tolist()
    return recommended_products

# Define function for displaying similar users and their purchase to justify recommendation 

def plot_user_user(user_id, similar_users, recommendations):

    G = nx.DiGraph()

    # target customer in center
    G.add_node(user_id, label="Target Customer", node_color="lightblue")

    # similar users
    for similar_user in similar_users:
        G.add_node(similar_user, label=f"Customer {similar_user}", node_color="lightgreen")
        G.add_edge(user_id, similar_user)

    # recommended products
    for product in recommendations:
        G.add_node(product, label=f"Product: {product}", node_color="orange")
        for similar_user in similar_users:
            G.add_edge(similar_user, product)

    # positions for nodes
    pos = nx.spring_layout(G, seed=42)

    # create edges
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=1, color='#888'),
        hoverinfo='none',
        mode='lines'
    )

    # create nodes
    node_x = []
    node_y = []
    node_labels = []
    node_colors = []
    for node in G.nodes(data=True):
        x, y = pos[node[0]]
        node_x.append(x)
        node_y.append(y)
        node_labels.append(node[1].get('label', str(node[0])))
        node_colors.append(node[1].get('node_color', "lightblue"))

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers+text',
        text=node_labels,
        textposition="top center",
        marker=dict(
            size=20,
            color=node_colors,
            line=dict(width=2, color='black')
        ),
        hoverinfo='text'
    )

    # create figure
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='Customer-Based Recommendation Graph',
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=0, l=0, r=0, t=40),
                        xaxis=dict(showgrid=False, zeroline=False),
                        yaxis=dict(showgrid=False, zeroline=False))
                    )

    return fig

# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------
# Data Prep for recomendation item-item
# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------


product_recommendation_df = df3[["CustomerID", "Product_Description"]]
product_recommendation_df = product_recommendation_df.dropna()

# user-item matrix
customer_product_matrix = product_recommendation_df.pivot_table(
    index='CustomerID',
    columns='Product_Description',
    aggfunc=lambda x: True,
    fill_value=False
)

# Reset the index and drop customer id feature
customer_product_matrix = customer_product_matrix.reset_index(drop=True)

# frequent item sets and association rules
frequent_itemsets_ap = apriori(customer_product_matrix, min_support=0.3, use_colnames=True)
rules_ap = association_rules(frequent_itemsets_ap, metric="confidence", min_threshold=0.8)

# function based on products (MBA)
def recommend_products(rules, product_basket):
    recommendations = set()
    for _, rule in rules.iterrows():
        antecedents = set(rule['antecedents'])
        consequents = set(rule['consequents'])
        
        if antecedents.issubset(product_basket):
            recommendations.update(consequents - set(product_basket))
    
    return list(recommendations)

# Function to display network of the most related items
def plot_rule_graph(rules, product_basket):

    G = nx.DiGraph()

    # nodes and edges based on rules
    for _, rule in rules.iterrows():
        antecedents = ', '.join(list(rule['antecedents']))
        consequents = ', '.join(list(rule['consequents']))
        
        if set(rule['antecedents']).issubset(product_basket):
            G.add_edge(antecedents, consequents, weight=rule['lift'])

    # positions for the nodes
    pos = nx.spring_layout(G, seed=42)

    # create  edges
    edge_x = []
    edge_y = []
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=1, color='#888'),
        hoverinfo='none',
        mode='lines'
    )

    # create nodes
    node_x = []
    node_y = []
    text = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        text.append(node)

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=text,
        textposition="top center",
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            reversescale=True,
            color=[],
            size=15,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
            line_width=2
        )
    )

    # colours based on number of connections
    node_adjacencies = []
    for node, adjacencies in enumerate(G.adjacency()):
        node_adjacencies.append(len(adjacencies[1]))
    node_trace.marker.color = node_adjacencies

    # figure
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='Recommendation Rule Path',
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=0, l=0, r=0, t=40),
                        xaxis=dict(showgrid=False, zeroline=False),
                        yaxis=dict(showgrid=False, zeroline=False))
                    )

    return fig

# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------
# Initialise the dashboard
# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------

app = Dash(__name__, suppress_callback_exceptions=True)


# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------
# Dark/Bright mode properties
# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------

light_theme = {
    "background": "#FFFFFF",
    "text": "black",
    "sidebar_bg": "#F8F9FA",
    "button_bg": "#5d646b",
    "button_color": "#FFFFFF",
}

dark_theme = {
    "background": "#1E1E1E", # background right side
    "text": "white",
    "sidebar_bg": "#2D2D2D", # background left side
    "button_bg": "blue",     # switch button
    "button_color": "green",
}

# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------
# Dashboard layout
# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------

app.layout = html.Div(
    # Main page
    id="main-container",
    style={"width": "100vw", "height": "100vh"},
    children=[
        html.Div(
            style={
                "display": "flex",
                "flexDirection": "row",
                "width": "100%",
                "height": "100%",
            },
            children=[
                # Sidebar
                html.Div(
                    id="sidebar",
                    style={
                        "flex": "0 0 30%",
                        "height": "100%",
                        "borderRight": "1px solid lightgray",
                        "padding": "10px",
                        "overflowY": "auto",
                        "backgroundColor": light_theme["sidebar_bg"], # set flexible colour for datk mode
                        "color": light_theme["text"],
                    },
                    children=[
                        # Main tabs on the top of the left side
                        dcc.Tabs(
                            id="main-tabs",
                            value="customers-info",
                            children=[
                                dcc.Tab(label="Customer Info", value="customers-info"),
                                dcc.Tab(label="Recommendations", value="recommendations"),
                                dcc.Tab(label="Sales", value="sales"),
                            ],
                        ),
                        html.Div(id="main-tab-content", style={"padding": "20px"}),
                        # Dark Mode Toggle Button
                        html.Div(
                            style={"marginTop": "20px", "textAlign": "center"},
                            children=[
                                html.Button(
                                    "Dark/Light  Mode",
                                    id="toggle-dark-mode",
                                    n_clicks=0,
                                    style={
                                        "marginTop": "10px",
                                        "padding": "10px",
                                        "backgroundColor": light_theme["button_bg"],
                                        "color": light_theme["button_color"],
                                        "border": "none",
                                        "borderRadius": "5px",
                                        "cursor": "pointer",
                                    },
                                ),
                            ],
                        ),
                    ],
                ),
                # Main Content Area
                html.Div(
                    id="output-content",
                    style={
                        "flex": "1",
                        "height": "100%",
                        "padding": "20px",
                        "overflowY": "auto",
                        "backgroundColor": light_theme["background"],
                        "color": light_theme["text"],
                    },
                    children=[
                        dcc.Graph(id="plot"),
                    ],
                ),
            ],
        ),
    ],
)


# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------

# Callbacks

# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------



# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------
# Callback dark mode switch 
# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------

@app.callback(
    [
        Output("main-container", "style"),
        Output("sidebar", "style"),
        Output("output-content", "style"),
        Output("toggle-dark-mode", "style"),
    ],
    [Input("toggle-dark-mode", "n_clicks")],
)
def toggle_dark_mode(n_clicks):
    # Determine the theme based on toggle state
    theme = dark_theme if n_clicks % 2 == 1 else light_theme

    # Update styles for each section
    main_style = {"width": "100vw", "height": "100vh"}
    sidebar_style = {
        "flex": "0 0 30%",
        "height": "100%",
        "borderRight": "1px solid lightgray",
        "padding": "10px",
        "overflowY": "auto",
        "backgroundColor": theme["sidebar_bg"],
    }
    content_style = {
        "flex": "1",
        "height": "100%",
        "padding": "20px",
        "overflowY": "auto",
        "backgroundColor": theme["background"],
        "color": theme["text"],
    }
    button_style = {
        "margin": "10px",
        "padding": "10px",
        "backgroundColor": theme["button_bg"],
        "color": theme["button_color"],
        "border": "none",
        "borderRadius": "5px",
        "cursor": "pointer",
    }

    return main_style, sidebar_style, content_style, button_style

# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------
# Add 3 tabs with sub tubs (Customer info, Recomendations and Sales)
# Define Properties (Input: tabs IDs line 72, Output: function update_tab_content)
# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------

@app.callback(
    [Output("main-tab-content", "children"), Output("output-content", "children")],
    Input("main-tabs", "value"),
)
def update_tab_content(tab_name):
    if tab_name == "customers-info":
        # Left-Side Inputs
        inputs = html.Div([
            dcc.Tabs(
                id="customer-info-tabs",
                value="specific-customer",
                children=[
                    dcc.Tab(label="Specific Customer", value="specific-customer"),
                    dcc.Tab(label="Demographic Group", value="demographic-group"),
                ],
            ),
            html.Div(id="customer-info-inputs", style={"padding": "20px"}),
        ])

        # Right side Outputs
        outputs = html.Div([
            html.Div(id="customer-info-outputs"),
        ])

        return inputs, outputs

    elif tab_name == "recommendations":
        # Left side Inputs with Sub-Tabs
        inputs = html.Div([
            dcc.Tabs(
                id="recommendation-tabs",
                value="based-on-customer",
                children=[
                    dcc.Tab(label="Based on Customer", value="based-on-customer"),
                    dcc.Tab(label="Based on Product", value="based-on-product"),
                ],
            ),
            html.Div(id="recommendation-inputs", style={"padding": "20px"}),
        ])

        # Right-Side Outputs
        outputs = html.Div([
            html.Div(id="recommendation-outputs"),
        ])

        return inputs, outputs

    elif tab_name == "sales":
    # Left side Inputs for Sales
        inputs = html.Div([
            html.H3("Sales Filters"),
            html.Label("Date Range:"),
            dcc.DatePickerRange(
                id="sales-date-picker",
                start_date=df["Transaction_Date"].min(),
                end_date=df["Transaction_Date"].max(),
            ),
            html.Br(),
            html.Label("Location:"),
            dcc.Dropdown(
                id="sales-location-dropdown",
                options=[{"label": loc, "value": loc} for loc in df["Location"].unique()],
                placeholder="Select a location",
            ),
            html.Label("Product Category:"),
            dcc.Dropdown(
                id="sales-category-dropdown",
                options=[{"label": cat, "value": cat} for cat in df["Product_Category"].unique()],
                placeholder="Select a product category",
            ),
            html.Label("Forecast Period (Days):"),
            dcc.Slider( 
                id="forecast-slider",
                min=1,
                max=30,
                step=1,
                value=7,
                marks={i: f"{i}d" for i in range(1, 31, 5)},
            ),
        ])
    
        # Right side Outputs
        outputs = html.Div([
            html.Div(id="sales-output"),
        ])
    
        return inputs, outputs

    return html.Div(), html.Div()

# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------
# Recommendation Sub-Tab Callback
# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------


@app.callback(
    [Output("recommendation-inputs", "children"), Output("recommendation-outputs", "children")],
    Input("recommendation-tabs", "value"),
)
def update_recommendation_tab_content(tab_name):
    if tab_name == "based-on-customer":
        inputs = html.Div([
            html.H4("Recommendations Based on Customer"),
            html.Label("Select or Enter Customer ID:"),
            dcc.Dropdown(
                id="recommendation-customer-dropdown",
                options=[{"label": str(customer_id), "value": customer_id} for customer_id in recommendation_df["CustomerID"].unique()],
                placeholder="Select a Customer ID",
            ),
            html.Br(),
            dcc.Input(id="recommendation-customer-id", type="text", placeholder="Or Enter Customer ID"),
            html.Button("Generate", id="recommendation-customer-button"),
        ])
    
        outputs = html.Div([
            html.Div(id="recommendation-customer-output"),
            dcc.Graph(id="recommendation-customer-graph"), 
        ])
    
        return inputs, outputs

    elif tab_name == "based-on-product":
        # Left Side Inputs
        inputs = html.Div([
            html.H4("Recommendations Based on Product"),
            html.Label("Select Products:"),
            dcc.Dropdown(
                id="recommendation-product-dropdown",
                options=[{"label": product, "value": product} for product in product_recommendation_df["Product_Description"].unique()],
                multi=True,
                placeholder="Select one or more products",
            ),
            html.Button("Generate", id="recommendation-product-button"),
        ])
        
        # Right Side Outputs
        outputs = html.Div([
            html.Div(id="recommendation-product-output"),
            dcc.Graph(id="recommendation-product-graph"), 
        ])
        
        return inputs, outputs

    return html.Div(), html.Div()


# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------
# User-User recomendation 
# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------

@app.callback(
    [Output("recommendation-customer-output", "children"),
     Output("recommendation-customer-graph", "figure")],
    [Input("recommendation-customer-button", "n_clicks")],
    [State("recommendation-customer-id", "value"),
     State("recommendation-customer-dropdown", "value")],
)
def generate_customer_recommendations(n_clicks, customer_id_text, customer_id_dropdown):
    customer_id = customer_id_dropdown or customer_id_text
    if n_clicks and customer_id:
        try:
            customer_id = int(customer_id)
        except ValueError:
            return f"Invalid Customer ID: {customer_id}. Please enter a valid numeric ID.", go.Figure()

        # recommendations and similat users
        recommendations = user_user(customer_id, num_recommendations=3)
        similar_users = user_item_matrix.loc[customer_id].nlargest(3).index.tolist()

        # Generate graph
        figure = plot_user_user(customer_id, similar_users, recommendations)

        return html.Ul([html.Li(product) for product in recommendations]), figure

    return "Enter a Customer ID and click Generate.", go.Figure()
    
# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------
# Item-Item recomendation 
# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------

@app.callback(
    [Output("recommendation-product-output", "children"),
     Output("recommendation-product-graph", "figure")], 
    [Input("recommendation-product-button", "n_clicks")],
    [State("recommendation-product-dropdown", "value")],
)
def generate_product_recommendations(n_clicks, product_basket):
    if n_clicks and product_basket:
        try:
            # Convert dropdown values
            product_basket = set(product_basket)
        except Exception as e:
            return f"Error processing selected products: {e}", go.Figure()

        # Generate recommendations
        recommendations = recommend_products(rules_ap, product_basket)
        
        # Generate graph for rules
        figure = plot_rule_graph(rules_ap, product_basket)

        if recommendations:
            return html.Ul([html.Li(product) for product in recommendations]), figure
        else:
            return "No recommendations found for the selected products.", go.Figure()

    return "Select products and click Generate.", go.Figure()



# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------
# Callbacks for Customer Info Sub-Tabs
# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------


@app.callback(
    [Output("customer-info-inputs", "children"), Output("customer-info-outputs", "children")],
    Input("customer-info-tabs", "value"),
)
def update_customer_info_content(tab_name):
    if tab_name == "specific-customer":
        # Left side Inputs
        inputs = html.Div([
            html.H4("Specific Customer Information"),
            html.Label("Select or Enter Customer ID:"),
            dcc.Dropdown(
                id="specific-customer-dropdown",
                options=[{"label": str(customer_id), "value": customer_id} for customer_id in df["CustomerID"].unique()],
                placeholder="Select a Customer ID",
            ),
            html.Br(),
            html.Label("Or Enter Customer ID:"),
            dcc.Input(id="specific-customer-id", type="text", placeholder="Customer ID"),
            html.Br(),
            html.Button("Search", id="search-button"),
        ])

        # Right aide Outputs
        outputs = html.Div([
            html.Div(id="specific-customer-info"),
            dcc.Graph(id="specific-customer-plot"),
        ])

        return inputs, outputs

    elif tab_name == "demographic-group":
    # Left side Inputs
        inputs = html.Div([
            html.H4("Demographic Group Analysis"),
            html.Label("Select Gender:"),
            dcc.Checklist(
                id="demographic-gender-checklist",
                options=[
                    {"label": "Male", "value": "M"},
                    {"label": "Female", "value": "F"}],
            ),
            html.Br(),
            html.Label("Select Location(s):"),
            dcc.Dropdown(
                id="demographic-location-dropdown",
                options=[{"label": loc, "value": loc} for loc in df["Location"].unique()],
                multi=True,
            ),
        ])
    
        # Right side Outputs (Bar plot + table)
        outputs = html.Div([
            dcc.Graph(id="demographic-category-plot"),  # Bar plor
            html.Div(id="demographic-analysis-output"),  # Table
        ])
    
        return inputs, outputs

    return html.Div(), html.Div()


# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------
# Callback for Specific Customer Information
# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------

@app.callback(
    [Output("specific-customer-info", "children"),
     Output("specific-customer-plot", "figure")],
    [Input("search-button", "n_clicks")],
    [State("specific-customer-id", "value"),
     State("specific-customer-dropdown", "value")], 
)
def fetch_specific_customer_info(n_clicks, customer_id_text, customer_id_dropdown):
    
    customer_id = customer_id_dropdown or customer_id_text  

    if n_clicks and customer_id:
        # Filter the data for the specified customer ID
        filtered_df = df[df["CustomerID"].astype(str).str.contains(str(customer_id))]
        
        if not filtered_df.empty:
            # Create a table for the customer transactions
            table = dash_table.DataTable(
                data=filtered_df[["CustomerID", "Transaction_ID", "Transaction_Date", "Product_Description", 
                                  "Product_Category", "Quantity", "Total_Spend"]].drop_duplicates().to_dict("records"),
                columns=[
                    {"name": "Customer ID", "id": "CustomerID"},
                    {"name": "Transaction ID", "id": "Transaction_ID"},
                    {"name": "Transaction Date", "id": "Transaction_Date"},
                    {"name": "Product Description", "id": "Product_Description"},
                    {"name": "Product Category", "id": "Product_Category"},
                    {"name": "Quantity", "id": "Quantity"},
                    {"name": "Total Spend ($)", "id": "Total_Spend"},
                ],
                style_table={"overflowX": "auto"},
                page_size=7, 
            )

            # data for  bar plot
            category_data = filtered_df.groupby("Product_Category")["Total_Spend"].sum().reset_index()

            #  bar plot
            bar_fig = px.bar(
                category_data,
                x="Product_Category",
                y="Total_Spend",
                title=f"Total Spend by Category for Customer {customer_id}",
                labels={"Product_Category": "Category", "Total_Spend": "Total Spend ($)"},
                color="Product_Category",
            )

            return table, bar_fig
        
        else:
            return html.Div("No matching customer found."), go.Figure()

    return html.Div("Enter a Customer ID and click Search."), go.Figure()



# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------
# Callback for Demographic Group Analysis
# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------

@app.callback(
    [Output("demographic-analysis-output", "children"), 
     Output("demographic-category-plot", "figure")],  
    [Input("demographic-gender-checklist", "value"), 
     Input("demographic-location-dropdown", "value")],
)
def analyze_demographic_group(selected_genders, selected_locations):
    filtered_df = df.copy()
    
    # filters based on gender
    if selected_genders:
        filtered_df = filtered_df[filtered_df["Gender"].isin(selected_genders)]
    
    # filters based on location
    if selected_locations:
        filtered_df = filtered_df[filtered_df["Location"].isin(selected_locations)]
    
    if not filtered_df.empty:
        # table for demographic analysis
        table = dash_table.DataTable(
            data=filtered_df[["Gender", "Location", "Product_Category", "Total_Spend"]]
                 .groupby(["Gender", "Location", "Product_Category"])
                 .sum()
                 .reset_index()
                 .to_dict("records"),
            columns=[
                {"name": "Gender", "id": "Gender"},
                {"name": "Location", "id": "Location"},
                {"name": "Product Category", "id": "Product_Category"},
                {"name": "Total Spend ($)", "id": "Total_Spend"},
            ],
            style_table={"overflowX": "auto"},
            page_size=7,
        )

        # data for the bar chart
        category_data = filtered_df.groupby("Product_Category")["Total_Spend"].sum().reset_index()

        # bar chart
        bar_fig = px.bar(
            category_data,
            x="Product_Category",
            y="Total_Spend",
            title="Total Spend by Category (Demographic Filter)",
            labels={"Product_Category": "Category", "Total_Spend": "Total Spend ($)"},
            color="Product_Category",
        )

        return table, bar_fig  # Return both the table and the bar chart figure

    else:
        # If no data matches the filters - return placeholders
        return html.Div("No data available for the selected filters."), go.Figure()
        
# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------
# Callback for sales
# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------

@app.callback(
    Output("sales-output", "children"),
    [Input("sales-date-picker", "start_date"), 
     Input("sales-date-picker", "end_date"),
     Input("sales-location-dropdown", "value"), 
     Input("sales-category-dropdown", "value"),
     Input("forecast-slider", "value")],  
)
def update_sales_output(start_date, end_date, location, category, forecast_period):
    filtered_df = df.copy()

    # Filter by date range
    if start_date and end_date:
        filtered_df = filtered_df[(filtered_df["Transaction_Date"] >= start_date) &
                                  (filtered_df["Transaction_Date"] <= end_date)]

    # Filter by location
    if location:
        filtered_df = filtered_df[filtered_df["Location"] == location]

    # Filter by product category
    if category:
        filtered_df = filtered_df[filtered_df["Product_Category"] == category]

    # If no data is available - return a message
    if filtered_df.empty:
        return html.Div("No sales data available for selected filters.")

    # Total sales by day for the line chart
    daily_sales = filtered_df.groupby("Transaction_Date")["Total_Spend"].sum().reset_index()

    # Ensure the data is sorted by date
    daily_sales.set_index("Transaction_Date", inplace=True)
    daily_sales.sort_index(inplace=True)

    # Fit the SARIMA model
    try:
        model = SARIMAX(
            daily_sales["Total_Spend"],
            order=(2, 1, 2), 
            seasonal_order=(2, 1, 2, 7),  
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        sarima_fit = model.fit(disp=False)

        # Forecast future sales
        forecast_index = pd.date_range(
            start=daily_sales.index[-1], 
            periods=forecast_period + 1, 
            freq="D"
        )[1:]  
        forecast_values = sarima_fit.get_forecast(steps=forecast_period).predicted_mean
        forecast_df = pd.DataFrame({"Transaction_Date": forecast_index, "Total_Spend": forecast_values})
    except Exception as e:
        return html.Div(f"Error fitting SARIMA model: {str(e)}")

    # Combine actuals and forecast for visualisation
    daily_sales.reset_index(inplace=True)
    combined_df = pd.concat([daily_sales, forecast_df])

    #  line chart
    line_fig = px.line(
        combined_df,
        x="Transaction_Date",
        y="Total_Spend",
        title="Total Sales Over Time",
        labels={"Transaction_Date": "Date", "Total_Spend": "Total Sales ($)"},
    )

    # forecast to the line chart with a different color
    line_fig.add_scatter(
        x=forecast_df["Transaction_Date"],
        y=forecast_df["Total_Spend"],
        mode="lines",
        name="Forecast",
        line=dict(color="red", dash="dot"),
    )

    #  map 
    state_sales = filtered_df.groupby("Location_Code")["Total_Spend"].sum().reset_index()
    map_fig = px.choropleth(
        state_sales,
        locations="Location_Code",
        locationmode="USA-states",
        color="Total_Spend",
        color_continuous_scale="Viridis",
        scope="usa",
        title="Total Sales by State",
        labels={"Total_Spend": "Total Sales ($)"},
    )

    # Return the map and line chart
    return html.Div([
        dcc.Graph(figure=map_fig), 
        dcc.Graph(figure=line_fig),  
        dash_table.DataTable(  
            data=filtered_df[["Transaction_ID", "Location", "Product_Category", "Total_Spend"]].to_dict("records"),
            columns=[
                {"name": "Transaction ID", "id": "Transaction_ID"},
                {"name": "Location", "id": "Location"},
                {"name": "Product Category", "id": "Product_Category"},
                {"name": "Total Spend ($)", "id": "Total_Spend"},
            ],
            style_table={"overflowX": "auto"},
            page_size=7,
        ),
    ])    
# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------
# Run the Dash app
# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------
if __name__ == "__main__":
    app.run_server(mode="external", debug=True)