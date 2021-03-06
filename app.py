"""
Create a scatterplot of market analysis residuals
      ~That's funny how this one started :)~

Created on Sun Jul  8 15:23:41 2018

@author: User
"""
# In[]:
################################# Switching to/from Ubuntu VPS ################

from sys import platform
import sys
import os

if platform == 'win32':
    homepath = "C:/Users/User/github/Ranch-Climate-Weather"
    statapath = "C:/Program Files (x86)/Stata15/Stata-64"
    dopath = "STATA/models/py_template_win.do"

    # Stata subprocess call - with call()
    def doStata(dopath, *params):
        cmd = [statapath, "/e", "do", dopath]
        for param in params:
            cmd.append(param)
        return(subprocess.call(cmd))

    # Set working directory
    os.chdir(homepath)
else:
    homepath = "/Ranch-Climate-Weather/"
    dopath = "STATA/models/py_template_linux.do"

    def doStata(dopath, which_df, formula):
        f = formula.split(" ")
        params = [f[i] for i in range(len(f))]
        empties = [" " for i in range(10 -len(f))]
        for e in empties:
            params.append(e)
        cmd =  ["stata", "-b", "do", dopath, which_df]
        for param in params:
            cmd.append(param)
        return(subprocess.call(cmd))

    # Set working directory
    os.chdir(homepath)
#############################################################################################################################
from functions import *

# In[]:
source_signal = ('["logweight L12.logweight winter1 spring1 summer1 fall1' +
                 'winter2 spring2 summer2 fall2 i.month", "500", "all",' +
                 '"all", "all", "all","all", "noaa"]')

# Some text items - Move all of these to text files somewhere
log = ''
description = ''
description_text = '''
##### 

This  is an exploratory  tool  used  in a research project aimed  at 
discovering a  climate signal in US cattle  market data. The research idea  is 
that,  given a history  of observations of a climate  variable, such as local 
or market-wide rainfall, it should be  possible to  explain some variance of
market factors such as  average cattle weights, sale counts,  or local price. 
Market data of the cattle production industry is  expected to exhibit a 
particularly strong climate signal compared to that of irrigated  grain 
production. The reason for this tool  is to utilize the power of a statistical
program called STATA, run as a subprocess routine in the background, for 
econometric modeling of a market dataset while taking  advantage of certain 
Python  libraries to  quickly 
observe spatial and temporal  trends of model predictions and residuals
following  specification. It  was designed specifically for researchers 
at the  University of  Colorado  Boulder to  work collaboratively  with 
researchers from other institutions on  this  problem. This  product is 
a test version, so please be patient of bugs or non-responsiveness.  It 
should  improve  over time in terms of  complexity and  functionality, as
well as in general performance. 

This tool uses cattle auction reports from 141 live auctions throughout
the central US that were collected between 2002 and 2017 by  the USDA's 
Agricultural Marketing  Service.  Monthly aggregations of  these market 
reports are associated with indexed precpitation values derived  from a 
product of NOAA's Climate  Prediction Center called  the Unified Gauge-
based Analysis of  Precipitation  over  CONUS. Precipitation  variables 
represent the  average index  value  for each location and time periods 
with one of four market radii: 100, 300, 500, and 700 km. These monthly 
variables are available for the month of sale ('t0'), as a lagged value 
of precipitation from 2 years (24 months) prior  to  sale ("t1 - t24"), 
as the precipitation value of  specific prior months for 2  years prior 
to sale  ("Jan1 - Dec2"), or as  an aggregated seasonal  value of prior 
(or current) seasons (for example, "Spr1" for the  precipitation values 
of  the  spring  preceeding  sale and "Spr2" for  values of  the spring 
before that). For seasonal precipitation values that fall in the season 
of the observed sale,  the  suffix  "1"  corresponds  to  precipitation 
values  within  the current  season up to  the point  of sale,  and the 
suffix "2"  corresponds to  the  respective season of the  prior  year. 

Please follow the popover text for instruction on how to use each 
element of the dashboard below. 

Earth Lab – CIRES at the University of Colorado Boulder
Author: Travis Williams
Email: Travis.Williams@colorado.edu
Date: 7-15-2018-2018
                        '''
                        
# Hover info text
stata_info = ("Enter a STATA formula here using the variable list to the " +
              "right. Variables are separated with a space, the "+
              "first element is the dependent variable and all subsequent " +
              "variables are independent. Use 'i.' as a prefix to "+
              "set fixed time effects. Here, use the variables 'year' for " +
              "year, 'month' for month, or 'time' for every monthly " + 
              "time-step. Use the prefix 'LN.' to specify a lagged " + 
              "predictor from N months back. This model uses a " + 
              "fixed-effects panel model algorithm with heterosketasticity " +
              "robust standard errors clustered by location. The template " +
              "in STATA appears as such: \n\nxtreg `formula', fe vce(robust)" +
              "\n\n with the variable `formula' corresponding "+
              "to the contents of this entry box")

map_info = ("This map displays the average value of the chosen output across the study period for each location. "+
            "Hover over each point to see the name of the auction and the value. Scroll to zoom in and out, "+
            "click and drag to pan, and hold control while clicking and dragging to change the viewing aspect. "+
            "Click on any single point to update the information graphs to the right and below.") 

pattern_info = ("This bar graph displays the average value of the chosen output at the chosen location for each month of the year.")
series_info = ("This bar chart shows a time series of individual values for the chosen output at the chosen location for each time step in the study period. "+
               "Gaps in the time series occur when the selected auction was either closed or that information was not reported by the AMS for any of a variety of reasons.")                               

# Data Set options
indexnames = [{'label' : 'CPC Unified Guage Base Precipitation', 
               'value' : 'noaa'},
              {'label' : 'Self-Calibrating Palmer Drought Severity Index',
               'value' : 'pdsisc'},
              {'label' : 'Palmer Drought Severity Index',
               'value' : 'pdsi'}]

# Map type options
maptypes = [{'label' : 'Light', 'value' : 'light'},
            {'label' : 'Dark', 'value' : 'dark'},
            {'label' : 'Basic', 'value' : 'basic'},
            {'label' : 'Outdoors', 'value' : 'outdoors'},
            {'label' : 'Satellite', 'value' : 'satellite'},
            {'label' : 'Satellite Streets', 'value' : 'satellite-streets'}]

# Total Variable Choices
variables = [{"label": '"local" - Auction name and location',"value":"locale"},
             {"label": '"class" - Class of cow',"value":"class"},
             {"label": '"grade" - Cow frame size',"value":"grade"},
             {"label": '"month" - Month',"value":"Month"},
             {"label": '"year" - Year',"value":"Year"},
             {"label": '"count" - Total monthly sales',"value":"count"},
             {"label": '"weight" - Average monthly cattle weight (lbs)',"value":"weight"},
             {"label": '"price" -  Average price ($/cwt)',"value":"price"},
             {"label": '"adj_price" - Price adjusted for inflation',"value":"adj_price"},
             {"label": '"adj_revenue" - Revenue from adjusted price',"value":"adj_revenue"},
             {"label": '"lat" - Latitude (decimal degrees)',"value":"lat"},
             {"label": '"lon" - Longitude (decimal degrees)',"value":"lon"},
             {"label": '"region" - (1 for North or 0 for South)',"value":"region"},
             {"label": '"t0" - Climate value for month of sale',"value":""},
             {"label": '"t1 - t24"  - Climate values 1 to 24 months prior to sale',"value":"t1 - t24"},
             {"label": '"jan1 - dec2"  - Months of climate values 1 to 2 years prior to sale',"value":"jan1 - dec2"},
             {"label": '"winter1 & winter2" - The 1st or 2nd winter prior to (of) sale',"value":"winter1 & winter2"},
             {"label": '"spring1 & spring2" - The 1st or 2nd spring prior to (of) sale',"value":"spring1 & spring2"},
             {"label": '"summer1 & summer2" - The 1st or 2nd summer prior to (of) sale',"value":"summer1 & summer2"},
             {"label": '"fall1 & fall2" - The first or second fall prior to (of) sale',"value":"fall1 & fall2"},]

# Create DASH application and server
app = dash.Dash(__name__)
app.css.append_css({'external_url': 'https://rawgit.com/WilliamsTravis/PRF-USDM/master/dash-stylesheet.css'})
server = app.server

# Create and initialize a cache for storing data - data pocket - not totally sure about this yet.
cache = Cache(config = {'CACHE_TYPE':'simple'})
cache.init_app(server)
# Create global chart template
mapbox_access_token = 'pk.eyJ1IjoidHJhdmlzc2l1cyIsImEiOiJjamZiaHh4b28waXNkMnptaWlwcHZvdzdoIn0.9pxpgXxyyhM6qEF_dcyjIQ'


# Stand in for model summary
rows = [{"Variable": "",
         "Coefficient": "",
         "P-Value": "",
         "Standard Error": ""}]

# Get dates 
date1 = datetime.date(2002,1,1).strftime('%b %Y')
date2 = datetime.date(2018,7,1).strftime('%b %Y')
dates = pd.DataFrame(pd.date_range(date1,date2,freq='MS'))
dates.columns = ["date"]
dates['date'] = pd.to_datetime(dates['date'])
dates['date'] = dates['date'].dt.strftime('%b %Y')
dates = np.array(dates['date'])


# Map Layout:
# Check this out! https://paulcbauer.shinyapps.io/plotlylayout/
layout = dict(
    autosize=True,
    height=500,
    font=dict(color='#CCCCCC'),
    titlefont=dict(color='#CCCCCC', size='20'),
    margin=dict(
        l=85,
        r=85,
        b=100,
        t=85,
        pad = 4
    ),
    bargap = 0.1,
    hovermode="closest",
    plot_bgcolor="#eee",
    paper_bgcolor="#04123b",
    legend=dict(font=dict(size=10), orientation='h'),
    title='<b>A Econometric Model of Cattle Market Climate Impacts</b>',
    mapbox=dict(#04123b
        accesstoken=mapbox_access_token,
        style="satellite-streets",#'light', 'basic', 'outdoors', 'satellite', or 'satellite-streets'
        center=dict(
            lon= -95.7,
            lat= 37.1
        ),
        zoom=2,
    )
)
# In[]:
        
# Page Layout

app.layout = html.Div([
                ################### Title ###############################
                html.Div([
                        
                        html.H1("An Econometric Model of Cattle Market Climate Impacts",
                                style = {'font-weight': 'bold'},
                                ),
                        html.Button(id = 'description_button',
                            children = 'Project Description (Click)',
                            title = description,
                            type='button',
                            style = {'margin-left':'35'}),
                        ]),
                html.Div(
                    [
                        dcc.Markdown(id = "description",
                                    children = description)
                    ],
                    style = {'text-align':'justify',
                             'margin-left':'35',
                             'margin-right': '50'}
                    ),
                        
                ################ hidden signal list  #################      
                html.Div(id='signal',
                         style={'display': 'none'}
                    ),
                html.Div(id='date_div',
                         children = dates,
                         style={'display': 'none'}
                    ), 
                
                # Break 
                html.Hr(style = {'height':'5',
                                 'border':'none',
                                 'color': '#e2e2e2',
                                 'background-color':'#4c4b4b'}),
                
                ################# Model Formula input ################
                html.Div([
                        
                        html.H2("Model Specification",
                                style = {'font-weight': 'bold'}),
                        # Formula Entry
                        html.Div([
                            html.H3("STATA Base Formula"),
                            dcc.Input(id = "formula",
                                      placeholder='y x1 x2 x3 ... ',
                                      type='text',
                                      value= "logweight L12.logweight winter1 spring1 summer1 fall1 winter2 spring2 summer2 fall2 i.month",
                                      style={'width': '100%'}),
                            html.Button(id='stata_info',
                                title = stata_info,
                                type='button',
                                n_clicks = 0,
                                children='Formula Info \uFE56 (Hover)',
#                                style = {'font-size':'8',
#                                         'padding': '1 1'},
                                ),
                                  ],
                                className = "seven columns",
                                style = {"margin-top":"10",
                                         'margin-left':'35',},

                                ),

    
                        # Variable Options
                        html.Div([
                            html.H3("Variable Choices"),
                            dcc.Dropdown(
                                    id = "variables",
                                    placeholder = "Click for dataset variable choices...",
                                    options = variables),
                                    ],
                                className = "four columns",
                                style = {"margin-top":"10"},
                                ),
                            ],
                        className = "row",
                        style = {"margin-bottom":"30"},

                        ),
        

                                        
                ############# All Filters ##################
                html.H3("Data Set Filters",
                        style = {
                                 'margin-left':'35',
                                 },),

                html.Div([
                    # Drought Index 
                    html.Div([
                            html.H5("Weather Index"),
                            dcc.Dropdown(
                                    id = 'index_choice',
                                    options = indexnames,
                                    value = "noaa"),
                                    ],
                                className = "three columns",
                            style = {
                                 'margin-left':'35',
                                 },
                        ),
                    # Market Radius 
                    html.Div([
                        html.H5("Market Radius"),
                        dcc.RadioItems(
                                id = "radii_filter",
                                options = [{'label':"100 km","value":"100"},
                                           {'label':"300 km","value":"300"},
                                           {'label':"500 km","value":"500"},
                                           {'label':"700 km","value":"700"}],
                                value = "500"
                                ),
                            ],
                          className = "one columns",

                        ),
                                
                    # Study Region            
                    html.Div([          
                        html.H5("Study Region"),
                        dcc.RadioItems(
                                id = "region_filter",
                                options = [{'label':"South","value":"0"},
                                           {'label':"North","value":"1"},
                                           {'label':"Both","value":"all"}],
                                value = "all"
                                ),   
                        ],
                      className = "one columns",
                    ),
                                
                    # Cow Class Filter
                    html.Div([
                        html.H5("Class Filter"),
                        dcc.RadioItems(
                                id = "class_filter",
                                options = [{'label':"Feeder Heifers","value": "Feeder_Heifers"},
                                           {'label':"Feeder Holstein Steers","value": "Feeder_Holstein_Steers"},
                                           {'label':"Feeder Steers","value": "Feeder_Steers"},
                                           {'label':"All","value":"all"}],
                                value = "all"
                                ),
                            ],
                          className = "one columns",
                        ),
                                
                    # Cow Frame size Filter
                    html.Div([
                        html.H5("Frame Size Filter"),
                        dcc.RadioItems(
                                id = "framesize_filter",
                                options = [{'label':"Small","value": "Small"},
                                           {'label':"Small and Medium","value": "Small_and_Medium"},
                                           {'label':"Medium","value": "Medium"},
                                           {'label':"Medium and Large","value": "Medium_and_Large"},
                                           {'label':"Large","value": "Large"},
                                           {'label':"All","value":"all"}],
                                value = "all"
                                ),
                            ],
                          className = "one columns",
                        ),
                            
                                                                
                    # Grade size Filter
                    html.Div([
                        html.H5("Muscle Grade Filter"),
                        dcc.RadioItems(
                                id = "grade_filter",
                                options = [{'label':"1","value": "1"},
                                           {'label':"1 to 2","value": "1_2"},
                                           {'label':"2","value": "2"},
                                           {'label':"2 to 3","value": "2_3"},
                                           {'label':"3","value": "3"},
                                           {'label':"4","value": "4"},
                                           {'label':"All","value":"all"}],
                                value = "all"
                                ),
                            ],
                          className = "one columns",
                        ),
                    
                    # Month of Sale Filter
                    html.Div([
                        html.H5("Month of Sale Filter"),
                        dcc.Dropdown(
                                id = "month_filter",
                                options = [{'label':"Jan","value": "1"},
                                           {'label':"Feb","value": "2"},
                                           {'label':"Mar","value": "3"},
                                           {'label':"Apr","value": "4"},
                                           {'label':"May","value": "5"},
                                           {'label':"June","value":"6"},
                                           {'label':"July","value":"7"},
                                           {'label':"Aug","value": "8"},
                                           {'label':"Sep","value": "9"},
                                           {'label':"Oct","value": "10"},
                                           {'label':"Nov","value": "11"},
                                           {'label':"Dec","value": "12"},
                                           {'label':"All","value": "all"}],
                                value = "all",
                                multi = True,
                                ),
                            ],
                          className = "one columns",
                        ),
                                
                # Submit and summary buttons
                    html.Div([
                            html.Button(id='submit', 
                                        type='submit',
                                        n_clicks = 0, 
                                        children='submit',
                                        style = {'font-weight':'bold'}
                                             
                                       ),
      
                        ],
                        className = "twelve columns",
                        style = {
#                               'text-align':'justify',
                                 'margin-left':'35',
                                 'margin-right':'150',
#                                 'margin-bottom':'75'
                                 "margin-bottom":"10"
                                 },
                    ),
                                
                                
                    ],
                  className = "row",
                ),
                
                # Break 
                html.Hr(style = {'height':'5',
                                 'border':'none',
                                 'color': '#e2e2e2',
                                 'background-color':'#4c4b4b'}),
                                 
                ############## Display Model Dianostics ########################
                # Summary Table
                html.H2("Model Summary",
                        style = {'font-weight':'bold',
                                 }),
                html.Div([
                        html.Button(id = "summary_button",
                                        title = "Click for model summary",
                                        children = "Model Summary Table (click)"
                                        ),

                        html.P(id="model_fit"),
                        html.P(id="model_n"),
                        html.P(id="model_constant"),
#                        html.P(id="model_se"),
                        dt.DataTable(
                             rows = rows,
                             id = "summary",
                             editable=False,
                             filterable=True,
                             sortable=True,
                             row_selectable=True,
    #                             min_width = 1655,
                             ),

                        ],
                    style = {'text-align':'justify',
                             'margin-left':'35',
                             'margin-right':'750',
                             'margin-bottom':'75'}
                        ),
                    
                    # Toggle log file display
                    html.Div([
                        html.H2("Stata Log file",
                                style = {'font-weight': 'bold'},
                                ),
                        html.Button(id = 'log_button',
                            children = 'Process Information (Click)',
                            title = "Click this to display this most recent STATA log",
                            type='button',
                            style = {'margin-left':'35'}),
                        ]),
                html.Div(
                    [
#                        dcc.Markdown(id = "log_text",
#                                    children = log)
                    html.Iframe(id = 'log_text',
                                srcDoc='',
                                style = {'width':'50%',
                                         'height':'500%'})

                    ],
                    style = {'text-align':'justify',
                             'margin-left':'35',
                             'margin-right': '50'}
                    ),
                       
                # Break 
                html.Hr(style = {'height':'5',
                                 'border':'none',
                                 'color': '#e2e2e2',
                                 'background-color':'#4c4b4b'}),

                ######################### Outputs #############################
                ############## Map Selection ###############
                html.H2("Model Outputs",
                        style = {'font-weight':'bold'}),
                html.Div([
                    html.Div([  
                        html.H5("Map Type"),
                            dcc.Dropdown(
                                    id = "map_type",
                                    value = "light",
                                    options = maptypes #'light', 'dark','basic', 'outdoors', 'satellite', or 'satellite-streets'   
                                ),
                            ],
                            style = {"margin-left":"35"},
                            className = "two columns",
                            ),
                    ],
#                    className = "row",
#                    style = {"margin-left":"35"},
                    ),
                        
                html.Div([
                    # Output type            
                    html.Div([   
                        html.H5("Output Type"),
                        dcc.Dropdown(
                                id = "output_type",
                                options = [{'label':"Predictions","value":"predictions_u"},
                                           {'label':"Observations","value":"observations"},
                                           {'label':"Residuals","value":"residuals_u"}],
                                value = "residuals_u",
                                multi = False
                                ),
                            ],
                          className = "two columns",
                          ),
                    ],
                    className = "row",
                    style = {"margin-left":"35"},

                    ),
                        
                ##########################################################
                html.Div([
                    # Background map selectioon
                    # The Map        
                    html.Div([
                            dcc.Graph(id = "main_graph"),  
                            html.Button("Map Info",
                                        title = map_info
                                        ),

    
                             ],
                            className = "six columns",
                            style={
                                   "margin-left":"35",
                                   'padding-right':'0',
                                   "margin-right":"0",
                                   "border-right":"0"
                                   },
                            ),
                    # Seasonal trend graph
                    html.Div([
                            dcc.Graph(id = "trend_graph"), # The seasonal trend graph
                            html.Button("Monthly Pattern Info",
                                        title = pattern_info
                                        ),
                            ],
                            className = "six columns",
                            style={
                                    'width':'47%',
                                   'float':'right',
                                   'padding-left':'0',
                                   "margin-left":"0",
                                   "border-left":"0"
                                   },
                            ),
                ],
                className = "row"
                ),
                            
                # The Time Series Graph
                html.Div([
                        dcc.Graph(id = "timeseries"),
                        html.Button("Time Series Info",
                                        title = series_info
                                        ),
                         ],
                        className = "twelve columns",
                        style = {'margin-top':'10',
                                 "margin-left":"35"
                                 }
                        ),

                ])

# In[]:
###############################################################################
######################### Create Cache ########################################
###############################################################################
@cache.memoize()
# Next step, create a numpy storage unit to store previously loaded arrays
def global_store(signal):
    
    # Unjsonify the signal (Back to original datatypes)
    # signal = source_signal
    signal = json.loads(signal)    
    formula = signal[0]
    radii_filter = str(signal[1])
    print(radii_filter)
    y = str(formula.split(" ")[0])
    index_choice = signal[7]
    
    # Get list of filter names
    filter_vars = []
    possible_vars =  ['class', 'framesize', 'grade']
    class_filter = signal[3]
    if class_filter != "all":
        filter_vars.append("class")
    framesize_filter = signal[4]
    if framesize_filter != "all":
        filter_vars.append("framesize")
    grade_filter = signal[5]
    if grade_filter != "all":
        filter_vars.append("grade")
    region_filter = signal[2]
    if region_filter != "all":
        filter_vars.append("region")
    month_filter = str(signal[6])
    if month_filter != "all":
#        month_filter = month_filter[0]
        filter_vars.append("month")
    
    # Create list of filters, we need to build the formula
    filters = np.array([class_filter, framesize_filter, grade_filter,
                        region_filter,month_filter])
    pos = np.where(filters!="all")
    
    # Add filters to the formula if needed
    if len(filters[filters == "all"]) == 5: # Uses aggregated data frame
        which_df = ("data/tables/rmw/" + index_choice + "/" + index_choice + 
                    "_" + str(radii_filter) + "_standardized_central_agg.csv")
        
    else:
        # Choose which df to read in - this one needs to built first
        df = pd.read_csv("data/tables/rmw/" + index_choice +
                         "/" + index_choice + "_" +
                         str(radii_filter) + "_standardized_central.csv")
        which_df = "data/tables/rmw/py_temp.csv"
        df['region'] = df['region'].apply(str)
        
        # For checking for month filters and creating the df
        # we have df and a list of filters.
        def splitCheck(filter_vars,filters,pos, varnum,df):
           
            #Check for month and find its position
            if "," in filters[pos][varnum]:
                fltr = filters[pos][varnum].split(",")
            else:
                fltr = filters[pos][varnum]
                if type(fltr) is not list:
                    fltr = [fltr]
            df = df[df[filter_vars[varnum]].apply(str).isin(fltr)]
            return df
        
        # Check for a list of month filters, then filter everything
        for i in range(len(filter_vars)):
            df = splitCheck(filter_vars,filters,pos, i-1, df)
       
        # Use the arguments to determine how to group and aggregate 
        id_list = [['locale','date'],filter_vars]
        group_list = [l for sl in id_list for l in sl]
        df['total_count'] = df.groupby(group_list)['count'].transform("sum")
        df['total_weight'] = df['weight'] * df['count']
        df['total_weight'] = df.groupby(group_list)['total_weight'].transform("sum")
        df['adj_price'] = df.groupby(group_list)['adj_price'].transform("mean")
        df['price'] = df.groupby(group_list)['price'].transform("mean")
        df['adj_revenue'] = df['adj_price']/100 * df['total_count'] * df['weight']
        df['adj_revenue'] = df.groupby(group_list)['adj_revenue'].transform("sum")
        df['revenue'] = df['price']/100 * df['count'] * df['weight']
        df['revenue'] = df.groupby(group_list)['revenue'].transform("sum")
        
        # now unselect the non-grouping columns
        non_vars = list(set(possible_vars)-set(filter_vars))
        drop_vars = [["logweight","Unnamed: 0","count","weight"], non_vars]
        drop_vars = [l for sl in drop_vars for l in sl]
        df2 = df.drop(drop_vars,axis = 1).drop_duplicates()
        df2['weight'] = df2['total_weight'] / df2['total_count']
        df2['logweight'] = np.log(df2['weight'])
        df2['count'] = df2['total_count']
        df2 = df2.drop('total_count',axis = 1)
        print("############## Df2 columns: " + str(df2.columns))
        df2.to_csv(which_df)

    #   Finally, run STATA
    if platform == 'win32':
        doStata(dopath,formula, which_df, y)
    else:
        doStata(dopath,which_df, formula)
        
    # get path from dropdown and read csv
    model = pd.read_csv("STATA/outputs/py_temp/pyout.csv")
    locales = model['locale'].unique()
    ids = [int(i) for i in range(len(locales))]
    points = dict(zip(locales,ids))
    model['pointIndex'] = model['locale'].map(points)
    
    # create the model for each panel - watchout for new output types
    # ...
    
    return model

def retrieve_data(signal):
#    if str(type(signal)) == "<class 'NoneType'>":
#        signal = source_signal
    df = global_store(signal)
    return df

###############################################################################
########################### Get Signal ########################################
###############################################################################
# Store the data in the cache and hide the signal to activate it in the hidden div
@app.callback(Output('signal', 'children'),
              [Input('submit','n_clicks')],
              [State('formula','value'),
               State('radii_filter','value'),
               State('region_filter','value'),
               State('class_filter','value'),
               State('framesize_filter','value'),
               State('grade_filter','value'),
               State('month_filter','value'),
               State('index_choice','value')
               ])
def compute_value(n_clicks,formula,radii_filter,region_filter,
                  class_filter,framesize_filter,grade_filter,
                  month_filter, index_choice):
    print("############################ month_filter = " + 
          str(month_filter) + " type = " + str(type(month_filter)))
    if len(month_filter) == 0:
        month_filter = "all"
    if type(month_filter) is list and len(month_filter) >1:
        month_filter = ",".join(month_filter)
    elif type(month_filter) is list:
        month_filter = month_filter[0]
    else:
        month_filter = month_filter
    print("MONTH FILTER == " + str(month_filter))
    signal = json.dumps([formula, radii_filter, region_filter, class_filter,
                         framesize_filter, grade_filter, month_filter, 
                         index_choice])
    print(signal)
    return signal
# In[]
@app.callback(Output('summary', 'rows'),
              [Input('summary_button', 'n_clicks')])
def toggleTable(click):

    # Read in CSV of coefficients, standard errors, and model fit. 
    results = pd.read_csv("STATA/results/py_temp/py_result.csv")
    results.columns = ["Variable","Coefficient", "Standard Error", "P-Value"]
    results = results.drop(results.index[0])   
    rows = results.to_dict('RECORDS')

    if not click:
        click = 0
    if click%2 == 1:
        summary = rows
    else:
        summary = ""
    return summary    

@app.callback(Output('model_fit', 'children'),
              [Input('summary_button', 'n_clicks'),
               Input('summary','rows')])
def toggleFit(click,rows):
    if not click:
        click = 0
    if click%2 == 1:
        r2 = "Within R-squared: " + rows[-1].get('Coefficient') 
    else:
        r2 = ""
    return r2

@app.callback(Output('model_n', 'children'),
              [Input('summary_button', 'n_clicks'),
               Input('summary','rows')])
def toggleN(click,rows):
    if not click:
        click = 0
    if click%2 == 1:
        N = "Number of Observations: " + rows[-2].get('Coefficient') 
    else:
        N = ""
    return N

@app.callback(Output('model_constant', 'children'),
              [Input('summary_button', 'n_clicks'),
               Input('summary', 'rows')])
def toggleConstant(click, rows):
    if not click:
        click = 0
    if click%2 == 1:
        C = "Constant: " + rows[-3].get('Coefficient') 
    else:
        C = ""
    return C

@app.callback(Output('description', 'children'),
              [Input('description_button', 'n_clicks')])
def toggleDescription(click):
    if not click:
        click = 0
    if click%2 == 1:
        description = description_text
    else:
        description = ""
    return description

@app.callback(Output('log_text','srcDoc'),
              [Input('log_button','n_clicks')])
def buildLog(click):
    if platform == "win32":
        log_file = "py_template_win.log"
    else:
        log_file = "py_template_linux.log"
    log_text = "".join(open(log_file,"r").readlines())
    log_text = re.sub("\n","<br>",log_text)
    if not click:
        click = 0
    if click%2 == 1:
        log = log_text
    else:
        log = ""
    return log 

@app.callback(Output('log_text','style'),
              [Input('log_button','n_clicks')])
def toggleLog(click):
    if not click:
        click = 0
    if click%2 == 1:
        style = {'width':'50%',
                'height':'500%'}
    else:
        style = {'display':'none'}
    return style 


# In[]              
@app.callback(Output("main_graph","figure"),
              [Input("signal","children"),
               Input("map_type","value"),
               Input("output_type","value")])
def makeMap(signal,map_type,output_type):
    
    # Get data
    model = retrieve_data(signal)
    
    # De-jsonify signal
    signal = json.loads(signal)
    # signal = source_signal
    
    # Extract arguments
    formula = signal[0]

    # Get the dependent variable
    dependent = str(formula.split(" ")[0])
    
    # Check which output
    if output_type == "observations":
        output_type = dependent

    
    # Correct for Fixed effect 
    #The original observations minus the demeaned model predictions plus the fixed effect
        # Be careful if we apply other transformations...or no transformation to the dependent variable
    if output_type == "residuals_u":
#        model['residuals_u'] = np.exp(model[dependent]) - np.exp(model['predictions_u'])  
        model['residuals_u'] = model['predictions_u'] - model[dependent] 
        output_print = "Mean Absolute Residuals - "  + dependent
    elif output_type == "predictions_u":
        output_print = "Predictions - " + dependent
    else:
        output_print = "Observations - " + dependent
        
    # Calculate Mean Absolute values or sum if counts
    if dependent is not "count":
        mar = model.groupby(['locale'])[output_type,'lon','lat'].apply(lambda x: np.mean(abs(x)))
#        print("##############!! Output_type: " + str(mar[dependent][0]))
    else:
        mar = model.groupby(['locale'])[output_type,'lon','lat'].sum()
#        print("##############!! Output_type: " + str(mar[dependent][0]))

    mar['lon'] = -1*mar['lon'] #lazy here
    mar = pd.DataFrame(mar)
    mar['locale'] = mar.index
    

    mar['text'] = mar['locale'] +'<br>' + mar[output_type].round(3).apply(str) 
    
    # Make Color Scale
    colorscale = [[0, 'rgb(2, 0, 68)'], [0.25, 'rgb(17, 123, 215)'],# Make darker (pretty sure this one)
                    [0.35, 'rgb(37, 180, 167)'], [0.45, 'rgb(134, 191, 118)'],
                    [0.6, 'rgb(249, 210, 41)'], [1.0, 'rgb(255, 249, 0)']] 
    
    # Set up scatterplot
    data = [
            dict(
                type = "scattermapbox",
                lon = mar['lon'],
                lat = mar['lat'],
                mode = 'markers',
                text = mar['text'],
                hoverinfo = 'text',
                marker = dict(
                        colorscale = colorscale,
                        color = mar[output_type],
                        opacity = .85,
                        size = 10,
                        colorbar = dict(
                                    textposition = "auto",
                                    orientation = "h",
                                    font = dict(size = 15)
                                    )
                                )
                            )
                        ]
    layout['title'] = "<b>" + output_print + "</b>"
    layout['mapbox'] = dict(
        accesstoken=mapbox_access_token,
        style= map_type, #'light', 'basic', 'outdoors', 'satellite', or 'satellite-streets'
        center=dict(
            lon= -95.7,
            lat= 37.1
        ),
        zoom=2,
    )
        
    figure = dict(data=data, layout=layout)
    return figure

# In[]:
@app.callback(Output("trend_graph","figure"),
              [Input("signal","children"),
               Input("output_type","value"),
               Input("main_graph","clickData")])    
def makeTrend(signal,output_type, clickData):
    # Before any click is made
    if clickData is None:
        pointIndex = 41
    else:
        pointIndex = clickData['points'][0]['pointIndex']

    # Get data
    model = retrieve_data(signal)
    
    # De-jsonify signal
    signal = json.loads(signal)
    # signal = source_signal
    
    print(str(output_type))
    
    # Get the dependent variable
    formula = signal[0]
    
    print(formula)
    dependent = str(formula.split(" ")[0])
    print(dependent)
    
    # Check which output
    if output_type == "observations":
        output_type = dependent
        
    # Correct for Fixed effect 
    #The original observations minus the demeaned model predictions plus the fixed effect
        # Be careful if we apply other transformations...or no transformation to the dependent variable

    model['residuals_u'] = model['predictions_u'] - model[dependent] 
    
    if output_type == "residuals_u":
        output_print = "Residuals "
    elif output_type == "predictions_u":
        output_print = "Predictions"
    else:
        output_print = "Observations"
    
#    model_nout = model[model['locale'].str.contains("Video") != True]
    
    # Get timeseries of residuals at the clickData location
    df = model[model['pointIndex'] == pointIndex]
    location = str(df['locale'].unique()[0])
    
    # For the x axis and value matching
#    dates = df['date'] # For grouping
#    df['month'] = df['date'].apply(lambda x: x[-8:-5]) # Comes with warning but works, if something breaks in the future...
    months = {1:'Jan',
              2:'Feb',
              3:'Mar',
              4:'Apr',
              5:'May',
              6:'Jun',
              7:'Jul',
              8:'Aug',
              9:'Sep',
              10:'Oct',
              11:'Nov',
              12:'Dec'}
    colors = [
        "#050f51",#'darkblue',
        "#1131ff",#'blue',
        "#09bc45",#'somegreensishcolor',
        "#6cbc0a",#'yellowgreen',
        "#0d9e00",#'forestgreen',
        "#075e00",#'darkgreen',
        "#1ad10a",#'green',
        "#fff200",#'yellow',
        "#ff8c00",#'red-orange',
        "#b7a500",#'darkkhaki',
        "#9195ad", # Greyish,
        "#6a7dfc", #'differentblue'
        ]
    
    # Group by month and calculate mean residuals, positive or negative
    monthlydf = df.groupby(['month'])["predictions_u","residuals_u",dependent].apply(lambda x: np.mean(x))
    monthlydf = pd.DataFrame(monthlydf)
    monthlydf['month'] = monthlydf.index
    monthlydf.index = monthlydf.index.set_names(['index'])
    monthlydf['month_abbr'] = monthlydf['month'].map(months)
    
    # sort properly
#    monthlydf['sort'] = monthlydf['month'].apply(lambda x: months.get(x))
    monthlydf = monthlydf.sort_values(by = "month")
    
    # Get Array
    yaxis = monthlydf[output_type]
    print(type(yaxis))
    xaxis = monthlydf['month_abbr']
#    xaxis = [months.get(i) for i in range(1,13)]
    # Set conditional y-axis scale
    # For maximum
    if output_type == "residuals_u":
        minimum = np.nanmin(model[output_type])*.55
        maximum = np.nanmax(model[output_type])*.55
    
    else:
        minimum_r = np.nanmin(model[dependent])
        minimum_o = np.nanmin(model["predictions_u"])
        minimum = min(minimum_r, minimum_o)
    
        maximum_r = np.nanmax(model[dependent])
        maximum_o = np.nanmax(model["predictions_u"])
        maximum = max(maximum_r, maximum_o)

    print(maximum)
    

        
    # Create annotation - we can add summary statistics here if needed.
#    annotation = dict(
##            text = scale_message,
#            y=0.95,
#            x=0.95,
#            font = dict(size = 17,color = "#000"),
#            showarrow=False,
#            xref="paper",
#            yref="paper")
    
    # Create list of dictionaries for plotly
    data = [
            dict(
                type = "bar",
                marker = dict(
                            color = colors,
                            line = dict(width = 3.5, color = "#000000")
                            ),
                x = xaxis,
                y = yaxis
                )    
            ]
    print(str(type(data)))
    
    yaxis_spec = dict(
        title = "Average " + output_print + " Value",
        autorange=False,
        range=[minimum, maximum],
        type='linear'
        )
    print(str(type(yaxis)))

    xaxis_spec = dict(
            tickfont =dict(color='#CCCCCC', size='20')
        )
            
    print(str(type(xaxis)))

    layout['title'] = "<b>Monthly Patterns - <br>" + location + "</b>"  
    print(layout['title'])
    layout['yaxis'] = yaxis_spec
    layout['xaxis'] = xaxis_spec
#    layout['annotations'] = [annotation]

    
    figure = dict(data = data, layout = layout)
    return figure

    
    
    
# In[]:
@app.callback(Output("timeseries","figure"),
              [Input("signal","children"),
               Input("output_type","value"),
               Input("main_graph","clickData"),
               Input("date_div","children")])
def makeSeries(signal,output_type,clickData,dates):
    
    # Before any click is made
    if clickData is None:
        pointIndex = 41
    else:
        pointIndex = clickData['points'][0]['pointIndex']

    # Get data
    model = retrieve_data(signal)
    
    # De-jsonify signal
    signal = json.loads(signal)
    # signal = source_signal
    
    # Get the dependent variable
    formula = signal[0]
    dependent = str(formula.split(" ")[0])
    
    # Check which output
    if output_type == "observations":
        output_type = dependent
        
    # Correct for Fixed effect 
    #The original observations minus the demeaned model predictions plus the fixed effect
        # Be careful if we apply other transformations...or no transformation to the dependent variable
    if output_type == "residuals_u":
#        model['residuals_u'] = np.exp(model[dependent]) - np.exp(model['predictions_u'])  
        model['residuals_u'] = model['predictions_u'] - model[dependent] 

        output_print = "Residuals "
    elif output_type == "predictions_u":
        output_print = "Predictions"
    else:
        output_print = "Observations"

    if output_type == "residuals_u":
        minimum = np.nanmin(model[output_type])
        maximum = np.nanmax(model[output_type])
    else:
        minimum_r = np.nanmin(model[dependent])
        minimum_o = np.nanmin(model["predictions_u"])
        minimum = min(minimum_r, minimum_o)
    
        maximum_r = np.nanmax(model[dependent])
        maximum_o = np.nanmax(model["predictions_u"])
        maximum = max(maximum_r, maximum_o)

    # Get timeseries of residuals at the clickData location
    df = model[model['pointIndex'] == pointIndex]
    location = str(df['locale'].unique()[0])
    
    # Get residuals
    res = np.array(df[['date',output_type]])
    pos = np.where(np.isin(dates,np.array(df['date'])))
    dates2 = [[date,np.nan] for date in dates]
    i =0
    for p in pos[0]:
        dates2[p][1] = res[i][1]
        i +=1
    values = np.array([d[1] for d in dates2])
    colors = np.where(values > 0, "red", "blue")

    # Set up y-axis 
    yaxis = dict(
        title = output_print + " Time Series",
        autorange = False,
        range = [minimum,maximum],
        type = 'linear'
        )
    xaxis = dict(
        tickfont =dict(color='#CCCCCC', size='10')
        )
            
    # Create Bar graph
    data = [dict(
            type = "bar",
            marker = dict(color = colors),#line = dict(width = 1.25, color = "#000000")),
            x = dates,
            y = values
            )]
    
    layout['title'] = "<b>" + output_print + " Time Series - " + location + "</b>"
    layout['yaxis'] = yaxis
    layout['xaxis'] = xaxis
#    layout['title'] = str(clickData)
#    layout['titlefont']=dict(color='#CCCCCC', size='2'),
#    layout['margin']=dict(
#        l=100,
#        r=85,
#        b=85,
#        t=255,
#        pad = 4
#    ),
#    
    
    figure = dict(data = data, layout = layout)
    return figure
    
    # In[]:
if __name__ == "__main__":
    app.run_server()
