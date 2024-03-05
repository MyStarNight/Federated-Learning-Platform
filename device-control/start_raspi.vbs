Sub Main

Dim username, password
username = "pi"
password = "raspberry"

Dim Hosts(9)
hosts(0) = "192.168.3.2"
hosts(1) = "192.168.3.3"
hosts(2) = "192.168.3.4"
hosts(3) = "192.168.3.7"
hosts(4) = "192.168.3.8"
hosts(5) = "192.168.3.10"
hosts(6) = "192.168.3.11"
hosts(7) = "192.168.3.12"
hosts(8) = "192.168.3.13"
hosts(9) = "192.168.3.20"


For Each HostStr In Hosts
    xsh.Session.Open ("ssh://" & username & ":" & password & "@" & HostStr)
    If xsh.Session.Connected Then
	End If
Next

End Sub
