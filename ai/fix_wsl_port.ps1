# Run as Administrator in PowerShell

$wslIp = wsl hostname -I | ForEach-Object { $_.Trim().Split(' ')[0] }
Write-Host "WSL2 IP: $wslIp"

# Remove old rule if exists
netsh interface portproxy delete v4tov4 listenport=8000 listenaddress=0.0.0.0 2>$null

# Add new forwarding rule
netsh interface portproxy add v4tov4 listenport=8000 listenaddress=0.0.0.0 connectport=8000 connectaddress=$wslIp

# Open firewall
netsh advfirewall firewall delete rule name="ShowRoom AI" 2>$null
netsh advfirewall firewall add rule name="ShowRoom AI" dir=in action=allow protocol=TCP localport=8000

Write-Host ""
Write-Host "Done! Tablet connect to: http://192.168.100.62:8000"
Write-Host ""
Write-Host "Forwarding: 192.168.100.62:8000 -> $wslIp:8000"
