# Getting started

## Mac OSX Setup (similar process for UNIX servers)

1. Create a new conda environment (Python 3.7)
```
conda create -n freecyto python=3.7
conda activate freecyto
```

2. Install python dependencies
```
pip install numpy
pip install -r requirements.txt
```

3. Setup a firebase account (https://console.firebase.google.com/)
Obtain firebase configuration credentials, as well as a Google Cloud service account json file.
```
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/json
```

Configuration details:
```
var config = {
  apiKey: "[APIKEY]",
  authDomain: "[project-id].firebaseapp.com",
  databaseURL: "https://[project-id].firebaseio.com",
  projectId: "[project-id]",
  storageBucket: "[project-id].appspot.com",
  messagingSenderId: "[sender-id]",
};
```

4. Replace in /templates/*.html var config with your custom config in the step above.
• dashboard.html
• quick_vis.html
• upload.html
• user.html

5. Run the application
```
python wsgi.py
```

6. Go to localhost:1500 to view the application.
