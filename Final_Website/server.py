from flask import Flask, render_template, request, jsonify, redirect, url_for
import importlib
import sys
import os

app = Flask(__name__)

# Dictionary to store authentication modules
auth_modules = {}

def load_auth_module(auth_type):
    """Dynamically load authentication module"""
    try:
        if auth_type not in auth_modules:
            module_name = f"{auth_type}_login"
            if module_name in sys.modules:
                # Reload module if already imported
                importlib.reload(sys.modules[module_name])
            
            module = importlib.import_module(module_name)
            auth_modules[auth_type] = module
            print(f"Loaded {auth_type} authentication module")
        
        return auth_modules[auth_type]
    except ImportError as e:
        print(f"Error loading {auth_type} module: {e}")
        return None

@app.route('/')
def home():
    """Main landing page with login options"""
    return render_template('login_options.html')

@app.route('/login/<auth_type>')
def login_page(auth_type):
    """Route to specific login page"""
    if auth_type not in ['voice', 'fingerprint', 'face']:
        return "Invalid authentication type", 400
    
    # Load the authentication module
    auth_module = load_auth_module(auth_type)
    if not auth_module:
        return f"Authentication module for {auth_type} not found", 500
    
    return render_template(f'{auth_type}_login.html')

@app.route('/authenticate/<auth_type>', methods=['POST'])
def authenticate(auth_type):
    """Handle authentication request"""
    auth_module = load_auth_module(auth_type)
    if not auth_module:
        return jsonify({"error": f"{auth_type} authentication not available"}), 500
    
    try:
        # Call the authenticate function from the loaded module
        result = auth_module.authenticate(request)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/success')
def success():
    """Success page after authentication"""
    return "<h1>Login Successful!</h1><p>You have been authenticated successfully.</p>"

def create_self_signed_cert():
    """Create a self-signed certificate for HTTPS"""
    try:
        from cryptography import x509
        from cryptography.x509.oid import NameOID
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.hazmat.primitives import serialization
        import datetime
        import ipaddress
        
        # Generate private key
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )
        
        # Get local IP
        import socket
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        
        # Create certificate
        subject = issuer = x509.Name([
            x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
            x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "Local"),
            x509.NameAttribute(NameOID.LOCALITY_NAME, "Development"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Auth Server"),
            x509.NameAttribute(NameOID.COMMON_NAME, local_ip),
        ])
        
        cert = x509.CertificateBuilder().subject_name(
            subject
        ).issuer_name(
            issuer
        ).public_key(
            private_key.public_key()
        ).serial_number(
            x509.random_serial_number()
        ).not_valid_before(
            datetime.datetime.utcnow()
        ).not_valid_after(
            datetime.datetime.utcnow() + datetime.timedelta(days=365)
        ).add_extension(
            x509.SubjectAlternativeName([
                x509.DNSName("localhost"),
                x509.IPAddress(ipaddress.ip_address("127.0.0.1")),
                x509.IPAddress(ipaddress.ip_address(local_ip)),
            ]),
            critical=False,
        ).sign(private_key, hashes.SHA256())
        
        # Save certificate and key
        with open("cert.pem", "wb") as f:
            f.write(cert.public_bytes(serialization.Encoding.PEM))
        
        with open("key.pem", "wb") as f:
            f.write(private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            ))
        
        return True, local_ip
    except ImportError:
        return False, None

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    # Try to create self-signed certificate
    cert_created, local_ip = create_self_signed_cert()
    
    if cert_created:
        print("✅ Self-signed certificate created!")
        print("⚠️  You'll need to accept the security warning in your browser")
        print("\nStarting HTTPS authentication server...")
        print("Available authentication methods:")
        print(f"- Voice Login: https://{local_ip}:5000/login/voice")
        print(f"- Fingerprint Login: https://{local_ip}:5000/login/fingerprint") 
        print(f"- Face Login: https://{local_ip}:5000/login/face")
        print(f"- Local access: https://localhost:5000/")
        
        # Run with HTTPS
        app.run(debug=True, host='0.0.0.0', port=5000, 
                ssl_context=('cert.pem', 'key.pem'))
    else:
        print("⚠️  Could not create HTTPS certificate (cryptography library not found)")
        print("Installing required library...")
        print("Run: pip install cryptography")
        print("\nFalling back to HTTP mode...")
        print("Note: Camera access may not work over HTTP on some browsers")
        
        app.run(debug=True, host='0.0.0.0', port=5000)