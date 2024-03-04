# LP
from packages.data_utils import *
from packages.globals import *
from packages.baseline import *

# Dash
import dash
import dash_bootstrap_components as dbc
from dash import dash_table as dt
from dash import html, dcc
from dash.dependencies import Input, Output, State, ALL

import nltk

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

app = dash.Dash(
	__name__,
	external_stylesheets=[
	'https://maxcdn.bootstrapcdn.com/font-awesome/4.7.0/css/font-awesome.min.css',
	'/static/plotly.css',
	'/static/custom_search.css'
	],
	prevent_initial_callbacks=True,
	suppress_callback_exceptions=True
	)
app.title = 'Narrative Baseline'

# Files
dataset = "cuba_160"
query = read_query_search(dataset)
#sim_table = compute_sim_with_t(query, str(dataset))

# Previous action storage.
previous_actions = []


card = dbc.Card(
    [
    	dbc.Row(
    		dbc.CardHeader("Publication Date: X", style={"background-color": "#D3D3D3", "font-weight": "bold", "text-align": "center", "font-size": "14px"}),
        ),
        dbc.Row(
            [
                dbc.Col(
                	html.Div(
                		html.Img(
                        	src="/static/default.svg",
                        	style = {
                        	"position":"relative",
                        	"top":"5%",
                        	"left":"15%",
                        	"transform":"translate(50%, 50%)"
                        	}
                    	),
                    ),
                    className = "two columns",
                    align="center"
                ),
                dbc.Col(
                    dbc.CardBody(
                        [
                            html.H4("Narrative Timeline", className="card-title"),
                            html.P(
                                "Please generate the timeline to visualize the data set.",
                                className="card-text",
                            ),
                            html.Small(
                                "Placeholder Card",
                                className="card-text text-muted",
                            ),
                        ]
                    ),
                    className = "eight columns",
                    #style={"margin": "auto"}#, "vertical-align": "middle"}
                ),
            ],
            className="g-0 d-flex align-items-center",
            style={"border-style": "solid", "border-color": "#D3D3D3"},
            justify="center",
        )
    ],
    style={"maxWidth": "800px", "margin": "auto", "padding-top": "25px"}
)

app.layout = html.Div([
	dbc.Row(
		style={'display': 'flex', 'align-items': 'center', 'vertical-align': 'middle', 'border-bottom': '2px solid black'},
		children=[
		dcc.Dropdown(
			id="dataset-choice",
			options=[{'label':'Cuban Protests', 'value': 'cuba_160'}, {'label':'COVID-19', 'value': 'cv'}],
			optionHeight=50,
			style={'width': '150px', 'margin-right': '5px'},
			value='cuba_160',
			clearable=False),
		html.Button(className="map_btn",
					style={'background-image' : 'url("/static/load_icon.svg")'},
					title="Load Data Set", id='load-data-button'),
		html.Span("Query", style={'fontSize': 12, 'fontWeight': 'bold', 'margin-right': '5px'}),
		dcc.Input(
			id="search-input", value="", type="text", style={'width': '200px', 'margin-right': '5px'}
		),
        html.Button(className="map_btn", style={'background-image' : 'url("/static/search_highlight.svg")'},
            title="Search and Generate News Timeline", id='search-button'),
		html.Span("Max Results", style={'fontSize': 12, 'fontWeight': 'bold', 'margin-right': '5px'}),
		dcc.Input(
			id="K-input", value=50, type="number", style={'width': '75px', 'margin-right': '5px'}
		),
		dcc.Loading(
			id="loading",
			type="default",
			fullscreen=True,
			style={ 'backgroundColor': '#FFFFFF50'}, children=html.Div('', id="loading-output")
		)
	]),
	dbc.Row([
	html.Div(className='eight columns', children=[
		html.Div(id='results', children=[card])
	]),

	html.Div(className='four columns', children=[
		dcc.Tabs(id='tabs', value="0", children=[
			dcc.Tab(label='Event Details', value="0", children=[
				html.Div(style=styles['tab'], children=[
					html.Div(
						id='tap-node-id',
						children='-1',
						style={'display': 'none'}
					),
					html.Div(
						id='tap-node-story',
						style= {'fontSize': 14}
					),
					html.Div(
						id='tap-node-title',
						style= {'fontSize': 14}
					),
					html.Div(
						id='tap-node-text',
						style= {'fontSize': 14}
					),
				])
			]),
			dcc.Tab(label='Data Set', value="2", children=[
				html.Div(id='data-table-tab', style=styles['tab'], children=[
					html.Div('Data table with all the events from the current data set. You can search for specific events here. Marking an event here will make it the starting point of your map.'),
					html.Button("Clear Selection", id="clear-tbl"),
					dt.DataTable(
						id='data-tbl', data=query.to_dict('records'),
						style_data={
							'whiteSpace': 'normal',
							'height': 'auto',
							'textAlign': 'left'
						},
						style_header={
							'backgroundColor': 'rgb(230, 230, 230)',
							'fontWeight': 'bold',
							'textAlign': 'center'
						},
						filter_action='native',
						page_current=0,
						page_size=20,
						page_action='native',
						columns=[{"name": i, "id": i} for i in query.loc[:,['date','title']]]
					)
				])
			]),
			#dcc.Tab(label='Options', value="3", children=[
			#	html.Div(style=styles['tab'], children=[
			#		html.Label('Filter by Dates', style={'font-weight': 'bold'}),
			#		html.Label('Filter the data set using a date range.'),
			#		dcc.DatePickerRange(
			#			id='date-range',
			#			minimum_nights=1,
			#			clearable=True,
			#			#with_portal=True,
			#			start_date_placeholder_text="Start Date",
			#			end_date_placeholder_text="End Date",
			#			start_date=date(1990, 1, 1),
			#			end_date=date(2030, 1, 1),
			#			style = {'width': '100%',
			#					'marginRight': '15px',
			#					'padding': '10px 5px',
			#					'borderRadius': '4px'
			#					}
			#		)
			#	])
			#])
		]),
	])
	]),
	dcc.Download(id="download")
])


@app.callback([Output('results', 'children'),
			   Output('loading-output', 'children'),
			   Output('data-table-tab', 'children')
			   ],
			  [Input('search-button', 'n_clicks'),
			   Input('load-data-button', 'n_clicks'),
			   Input('search-input', 'n_submit')
			  ],
			  [State('search-input', 'value'),
			   State('K-input', 'value'),
			   State('dataset-choice', 'value'),
			   State('data-table-tab', 'children'),
			   #State('date-range', 'start_date'),
			   #State('date-range', 'end_date')
			  ], prevent_initial_call=True)
def interact_with_graph(search_button, load_data, search_input_enter, search_query, max_results,
						dataset, data_table_tab):#, start_date, end_date):#, rep_landmark_mode):
	ctx = dash.callback_context
	if not ctx.triggered:
		button_id = 'No clicks yet'
		return [search_results, '', data_table_tab]

	# If we are here this means we clicked a "classic" button!
	button_id = ctx.triggered[0]['prop_id'].split('.')[0]
	if button_id == 'search-button' or button_id == 'search-input':
		query = read_query_search(dataset)#, start_date, end_date)
		print("Data file has been read successfully.")
		sim_list = search_dataset(search_query, query, dataset)
		print("Relevance table has been computed.")
		if search_query.strip() == "": # If no search query, assume uniform relevance
			sim_list = np.ones_like(sim_list)
		query['similarity'] = sim_list
		query["similarity"] = pd.to_numeric(query["similarity"])
		if search_query.strip() == "":
			query_sorted = query.sort_values(['date'],ascending=False)
		else:
			n_groups = 20
			query['group'] = pd.qcut(query['similarity'], q=n_groups, labels=list(range(n_groups))) # 5% divisions.
			if len(query.index) >= max_results and max_results >= n_groups:
				n_head = round(max_results / (n_groups / 2), 0)
				query_sorted = query.sort_values(['group', 'date'],ascending=[False, False]).groupby('group').head(int(n_head))
			else:
				query_sorted = query.sort_values(['group', 'date'],ascending=[False, False])
		block = []
		block_count = 0
		search_results = []
		count_elements = 1

		for index, row in query_sorted.iterrows():
			contents = row['full_text']
			source = "/static/" + str(row['publication']) + ".svg"
			#print(count_elements)
			if count_elements > max_results:
				print("Exceeded max results.")
				if len(block) > 0:
					#print("Adding final block.")
					block_count += 1
					if block_count == 1:
						search_results.append(html.Li(id={'type': 'pagination-contents', 'index': block_count}, children=block, style={"display": "block"}))
					else:
						search_results.append(html.Li(id={'type': 'pagination-contents', 'index': block_count}, children=block, style={"display": "none"}))
				break # Stop after max_results.
			if search_query.strip() == "":
				relevance_str = ""
			else:
				relevance_str = "  Relevance: " + str(round(row['similarity'] * 100, 2)) + "%"# + " - Group: " + str(row['group'])
			card = dbc.Card(
				    [
				    	dbc.CardHeader("Publication Date: " + str(row['date']),
				    		style={"background-color": "#D3D3D3", "font-weight": "bold", "text-align": "center", "font-size": "14px"}),
				        dbc.Row(
				            [
				                dbc.Col(
				                	html.Div(
				                		html.Img(
				                        	src=source,
				                        	style = {
				                        	"position":"relative",
				                        	"top":"5%",
				                        	"left":"15%",
				                        	"transform":"translate(50%, 50%)"
				                        	}
				                    	),
				                    ),
				                    className = "two columns",
				                    align="center"
				                ),
				                dbc.Col(
				                    dbc.CardBody(
				                        [
				                            html.H4(row['title'], className="card-title"),
				                            html.P(
				                                (contents[:300] + '...') if len(contents) > 300 else contents,
				                                className="card-text",
				                            ),
				                            dbc.Button("Read More", color="primary", id={'type': 'btn-read-more', 'index': index}, className="read_more_btn"),
				                            html.Small(
				                                relevance_str,
				                                className="card-text text-muted",
				                            ),
				                        ]
				                    ),
				                    className = "nine columns",
				                    style={"padding-bottom": "5px"}#, "vertical-align": "middle"}
				                ),
				            ],
				            className="g-0 d-flex align-items-center",
				            style={"border-style": "solid", "border-color": "#D3D3D3"},
				            justify="center",
				        )
				    ],
				    style={"maxWidth": "800px", "margin": "auto", "padding-top": "25px"}
				)

			count_elements += 1
			block.append(card)
			if len(block) == 10:
				#print("Added block.")
				block_count += 1
				if block_count == 1:
					search_results.append(html.Li(id={'type': 'pagination-contents', 'index': block_count}, children=block.copy(), style={"display": "block"}))
				else:
					search_results.append(html.Li(id={'type': 'pagination-contents', 'index': block_count}, children=block.copy(), style={"display": "none"}))
				block = []


		pagination = html.Div(
 		   dbc.Pagination(max_value=block_count, id="pagination", className="page_menu"),
		)
		search_results = [pagination, html.Ul(children=search_results, className="pagination")]
		status_msg = "Search Results Generated"

		return [search_results, html.P([status_msg]), data_table_tab]
	elif button_id == 'load-data-button':
		query = read_query_search(dataset)#, start_date, end_date)
		new_table = [html.Div('Data table with all the events from the current data set. You can search for specific events here.'),
					dt.DataTable(
						id='data-tbl', data=query.to_dict('records'),
						style_data={
							'whiteSpace': 'normal',
							'height': 'auto',
							'textAlign': 'left'
						},
						style_header={
							'backgroundColor': 'rgb(230, 230, 230)',
							'fontWeight': 'bold',
							'textAlign': 'center'
						},
						filter_action='native',
						page_current=0,
						page_size=20,
						page_action='native',
						columns=[{"name": i, "id": i} for i in query.loc[:,['date','title']]]),
					]
		card = dbc.Card(
				[
					dbc.Row(
						dbc.CardHeader("Publication Date: X", style={"background-color": "#D3D3D3", "font-weight": "bold", "text-align": "center", "font-size": "14px"}),
				    ),
				    dbc.Row(
				        [
				            dbc.Col(
				            	html.Div(
				            		html.Img(
				                    	src="/static/default.svg",
				                    	style = {
				                    	"position":"relative",
				                    	"top":"5%",
				                    	"left":"15%",
				                    	"transform":"translate(50%, 50%)"
				                    	}
				                	),
				                ),
				                className = "two columns",
				                align="center"
				            ),
				            dbc.Col(
				                dbc.CardBody(
				                    [
				                        html.H4("Narrative Timeline", className="card-title"),
				                        html.P(
				                            "Please generate the timeline to visualize the data set.",
				                            className="card-text",
				                        ),
				                        html.Small(
				                            "Placeholder Card",
				                            className="card-text text-muted",
				                        ),
				                    ]
				                ),
				                className = "eight columns",
				                #style={"margin": "auto"}#, "vertical-align": "middle"}
				            ),
				        ],
				        className="g-0 d-flex align-items-center",
				        style={"border-style": "solid", "border-color": "#D3D3D3"},
				        justify="center",
				    )
				],
				style={"maxWidth": "800px", "margin": "auto", "padding-top": "25px"}
				)
		previous_actions.clear()
		status_msg = "Data Set Loaded"
		return [card, html.P([status_msg]), new_table]
	return [search_results, html.P([status_msg]), data_table_tab]



@app.callback(
	Output("data-tbl", "selected_rows"),
	Input("clear-tbl", "n_clicks"),
)
def clear(n_clicks):
	return []



@app.callback(
    Output({'type': 'pagination-contents', 'index': ALL}, 'style'),
    Input('pagination', 'active_page'),
    State({'type': 'pagination-contents', 'index': ALL}, 'style')
)
def display_output(active_page, styles):
	# Iterate over all outputs.
	# Change style of all IDs to hidden.
	# Find correct ID matching page and show.
    for idx in range(len(styles)):
    	styles[idx] = {"display": "none"}
    	if idx == active_page - 1:
    		styles[idx] = {"display": "block"}
    return styles


@app.callback(
    [Output('tap-node-id', 'children'),
	 Output('tap-node-story', 'children'),
	 Output('tap-node-title', 'children'),
	 Output('tap-node-text', 'children')],
    Input({'type': 'btn-read-more', 'index': ALL}, 'n_clicks'),
    [State('dataset-choice', 'value'),
     #State('date-range', 'start_date'),
	 #State('date-range', 'end_date')
	 ]
)
def display_output(btn_clicks, dataset):#, start_date, end_date):
    ctx = dash.callback_context
    if not ctx.triggered:
        button_id = 'No clicks yet'
        return ["", "", "", ""]
    p = re.compile(r"(\"index\"):(\d+)")
    query = read_query(dataset, partial=True) #start_date, end_date
    trigger_node = list(dash.callback_context.triggered_prop_ids.keys())[0]
    m = p.search(trigger_node)
    node_id = m.group(2) # Second parenthesis, get element that triggered this.
    offset = min(query.id.astype('int'))
    node_row = query.iloc[int(node_id) - offset]
    string_list = node_row['full_text'].split(sep='\n')




    title = html.Div(node_row['title'], style={'fontSize': 16, 'fontWeight': 'bold'}, contentEditable=False)
    output_list = [html.Div(node_row['date'], style={'fontSize': 12, 'font-style': 'italic'}),
                   html.A(node_row['url'], href=node_row['url'], target='_blank', style={'fontSize': 12, 'font-style': 'italic'})]
    node_story = ""
    if len(string_list) == 1: # No natural split.
        splitter = re.compile(r'''((?:[^\."']|"[^"]*"|'[^']*'|\([^\)]*\))+)''')
        p_n = 4 # Split into paragraphs of 4 if necessary. Lots of bug fixes to handle U.S. case.
        split_list = splitter.split(string_list[0].strip())
        split_list_copy = []
        previous_us = ""
        for s in split_list:
            if s == "S":
                split_list_copy[-1] += "." + s # Special U.S. case
                previous_us = "S"
            elif s == ".":
                if previous_us == "S":
                    split_list_copy[-1] += s
                    previous_us = "S."
                continue # Do nothing
            else:
                if previous_us == "S.":
                    split_list_copy[-1] += " " + s
                    previous_us = ""
                else:
                    split_list_copy.append(s)
        split_list_p_n = [split_list_copy[i * p_n:(i + 1) * p_n] for i in range((len(split_list_copy) + p_n - 1) // p_n )]
        for s in split_list_p_n: # List of list of strings.
            # Concatenate and output.
            output_list += [html.P(". ".join(s) + ".", contentEditable=False)]    
    else:
        for s in string_list:
            if len(s.strip()) > 0:
                output_list += [html.P(s.strip(), contentEditable=False)]

    return [node_id, node_story, title, output_list]

if __name__ == "__main__":
	app.run(port=8051)
