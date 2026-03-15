from robot.api_server import app, init_hardware  
  
if __name__ == '__main__':  
    init_hardware()  
    app.run(host='0.0.0.0', port=5000, debug=False) 
