# Kickstarter 1
An app to predict the success of a notional kickstarter campaign using user defined inputs in an app created with streamlit. 

- load_data.ipynb: Notebook to read raw Kickstarter source files from web scraper ainto usable format and Exploratory Data Analysis
-  model.py: Select Key Numerical, Categorical, and Textual Features -- Run Tfidf vectorization on project description create new features that counts most common and unique phrases that appear in successful / failed campaigns and One Hot Encode the feature. 
-  app.py - Front End Application built using streamlit to prompt user for: 
  -  # of backers the user believes their campaign can reach
  -  whether the campaign is currently a spotlight campaign on kickstarter
  -  the length of the campaign
  -  Whether campaign is currently a staff pick
  -  What currency the campaign is raised in
  -  Text Description of the project
  -  Goal for kickstarter campaign in currency selected

IN WORK
