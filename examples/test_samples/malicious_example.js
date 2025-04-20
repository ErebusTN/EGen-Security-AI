// This is a test file with intentionally malicious-looking code for scanner testing
// DO NOT EXECUTE THIS CODE - FOR TESTING PURPOSES ONLY

// Obfuscated JavaScript example
var a123456789012345678901234567890 = "\x68\x74\x74\x70\x73\x3A\x2F\x2F\x6D\x61\x6C\x69\x63\x69\x6F\x75\x73\x2E\x65\x78\x61\x6D\x70\x6C\x65\x2E\x63\x6F\x6D";

// Suspicious eval with encoded content
function loadPayload() {
  eval(atob("YWxlcnQoImV4YW1wbGUgbWFsaWNpb3VzIGNvZGUiKTsK"));
}

// XSS pattern
var xssExample = "<script>alert(document.cookie)</script>";

// Suspicious URL pattern
var config = {
  endpoint: "/admin.php?id=1234567890123456",
  fallback: "/gate.php?q=setup"
};

// SQL Injection pattern
function unsafeQuery(input) {
  // Deliberately unsafe for demonstration
  return "SELECT * FROM users WHERE username = '" + input + "' OR 1=1--";
}

// Example usage (never actually execute this)
function maliciousDemo() {
  var input = "'OR 1=1;--";
  unsafeQuery(input);
  loadPayload();
  
  // This would be detected by the scanner
  document.getElementById("test").innerHTML = xssExample;
}

// Export for completeness
export { maliciousDemo }; 