# Run as Administrator in PowerShell
# Forwards port 8080 — ai/server.py now serves the whole site (index.html,
# assets, PDFs) AND the AI chat on this single port, so only one forward
# and one firewall rule are needed.

$wslIp = wsl hostname -I | ForEach-Object { $_.Trim().Split(' ')[0] }
Write-Host "WSL2 IP: $wslIp"

netsh interface portproxy delete v4tov4 listenport=8080 listenaddress=0.0.0.0 2>$null
netsh interface portproxy add v4tov4 listenport=8080 listenaddress=0.0.0.0 connectport=8080 connectaddress=$wslIp

netsh advfirewall firewall delete rule name="ShowRoom Site" 2>$null
netsh advfirewall firewall add rule name="ShowRoom Site" dir=in action=allow protocol=TCP localport=8080

Write-Host ""
Write-Host "Done! On the tablet, open: http://192.168.100.58:8080"
Write-Host ""
Write-Host "Forwarding: 192.168.100.58:8080 -> ${wslIp}:8080"
