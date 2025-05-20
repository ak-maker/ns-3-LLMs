<u><i>Still under active construction</i></u>

## Live demo For Scenario 2 using GPT-4o

🎥 [Watch Demo B1 & B2](https://youtu.be/gPpn4JDn7Kw)

🎥 [Watch Demo B3](https://youtu.be/QiSFq-fsGH8)

## **Complete implementation scripts for B. Scenario 2: Packet-level Simulation with ns-3**

### B1. LLM-Driven Automation

**Question:**

```text
Create a complete ns-3 simulation with a dynamic emitter class generating exponential interarrival times
(mean = 1.0 second) and using ChangeRate(double newRate) to adjust the mean to 0.5 seconds at 50 seconds;
connect a TimeProbe to the Interval trace source via log changes with TraceConnect function, and utilize
GnuplotHelper to create a PNG graph, while ensuring accurate trace source paths to prevent runtime errors.
```

**Script:**

```cpp
#include "ns3/core-module.h"
#include "ns3/stats-module.h"
#include "ns3/gnuplot-helper.h"

using namespace ns3;

class DynamicEmitter : public Object
{
public:
  static TypeId GetTypeId()
  {
    static TypeId tid = TypeId("DynamicEmitter")
      .SetParent<Object>()
      .AddConstructor<DynamicEmitter>()
      .AddTraceSource("Interval",
                      "Interarrival time between events",
                      MakeTraceSourceAccessor(&DynamicEmitter::m_interval),
                      "ns3::Time::TracedValueCallback");
    return tid;
  }

  DynamicEmitter() : m_var(CreateObject<ExponentialRandomVariable>()) {}

  void Start()
  {
    m_var->SetAttribute("Mean", DoubleValue(1.0));
    Emit();
  }

  void ChangeRate(double newRate)
  {
    m_var->SetAttribute("Mean", DoubleValue(newRate));
  }

private:
  void Emit()
  {
    Time interval = Seconds(m_var->GetValue());
    m_interval = interval;
    Simulator::Schedule(interval, &DynamicEmitter::Emit, this);
  }

  TracedValue<Time> m_interval;
  Ptr<ExponentialRandomVariable> m_var;
};

int main(int argc, char *argv[])
{
  CommandLine cmd;
  cmd.Parse(argc, argv);

  Time::SetResolution(Time::NS);

  Ptr<DynamicEmitter> emitter = CreateObject<DynamicEmitter>();
  Names::Add("/Names/DynamicEmitter", emitter);

  Ptr<TimeProbe> timeProbe = CreateObject<TimeProbe>();
  bool connected = timeProbe->ConnectByObject("Interval", emitter);
  NS_ASSERT(connected);

  GnuplotHelper plotHelper;
  plotHelper.ConfigurePlot("dynamic-emitter-interarrival-time",
                           "Emitter Interarrival vs. Time with Rate Change",
                           "Time (Seconds)",
                           "Interarrival Time (Seconds)",
                           "png");

  plotHelper.PlotProbe("ns3::TimeProbe",
                       "/Names/DynamicEmitter/Interval",
                       "Output",
                       "Dynamic Emitter Interarrival Time",
                       GnuplotAggregator::KEY_INSIDE);

  Simulator::Schedule(Seconds(0.0), &DynamicEmitter::Start, emitter);
  Simulator::Schedule(Seconds(50.0), &DynamicEmitter::ChangeRate, emitter, 0.5);

  Simulator::Stop(Seconds(100.0));
  Simulator::Run();
  Simulator::Destroy();

  return 0;
}

```

### B2. Network Simulation Refinement through Multi-Round Interaction

**Question:**

```text
What if we wanted to start with a mean interarrival time of 0.5 seconds, and at 50 seconds call ChangeRate
so the new mean becomes 0.4 seconds?
```

**Script:**

```cpp
#include "ns3/core-module.h"
#include "ns3/stats-module.h"
#include "ns3/gnuplot-helper.h"

using namespace ns3;

class DynamicEmitter : public Object
{
public:
  static TypeId GetTypeId()
  {
    static TypeId tid = TypeId("DynamicEmitter")
      .SetParent<Object>()
      .AddConstructor<DynamicEmitter>()
      .AddTraceSource("Interval",
                      "Interarrival time between events",
                      MakeTraceSourceAccessor(&DynamicEmitter::m_interval),
                      "ns3::Time::TracedValueCallback");
    return tid;
  }

  DynamicEmitter() : m_var(CreateObject<ExponentialRandomVariable>()) {}

  void Start()
  {
    m_var->SetAttribute("Mean", DoubleValue(0.5));
    Emit();
  }

  void ChangeRate(double newRate)
  {
    m_var->SetAttribute("Mean", DoubleValue(newRate));
  }

private:
  void Emit()
  {
    Time interval = Seconds(m_var->GetValue());
    m_interval = interval;
    Simulator::Schedule(interval, &DynamicEmitter::Emit, this);
  }

  TracedValue<Time> m_interval;
  Ptr<ExponentialRandomVariable> m_var;
};

int main(int argc, char *argv[])
{
  CommandLine cmd;
  cmd.Parse(argc, argv);

  Time::SetResolution(Time::NS);

  Ptr<DynamicEmitter> emitter = CreateObject<DynamicEmitter>();
  Names::Add("/Names/DynamicEmitter", emitter);

  Ptr<TimeProbe> timeProbe = CreateObject<TimeProbe>();
  bool connected = timeProbe->ConnectByObject("Interval", emitter);
  NS_ASSERT(connected);

  GnuplotHelper plotHelper;
  plotHelper.ConfigurePlot("dynamic-emitter-interarrival-time",
                           "Emitter Interarrival vs. Time with Rate Change",
                           "Time (Seconds)",
                           "Interarrival Time (Seconds)",
                           "png");

  plotHelper.PlotProbe("ns3::TimeProbe",
                       "/Names/DynamicEmitter/Interval",
                       "Output",
                       "Dynamic Emitter Interarrival Time",
                       GnuplotAggregator::KEY_INSIDE);

  Simulator::Schedule(Seconds(0.0), &DynamicEmitter::Start, emitter);
  Simulator::Schedule(Seconds(50.0), &DynamicEmitter::ChangeRate, emitter, 0.4);

  Simulator::Stop(Seconds(100.0));
  Simulator::Run();
  Simulator::Destroy();

  return 0;
}

```

### B3. Simulation Code Transition and Log Analysis

**Question:**

> Please note: During the actual interaction, the user only needs to provide the path to the C++ file requiring conversion. Subsequent formatting will be handled automatically for question input.

```text
Turn this ns-3 c++ script into its python version:

#include "ns3/applications-module.h"
#include "ns3/core-module.h"
#include "ns3/internet-module.h"
#include "ns3/network-module.h"
#include "ns3/point-to-point-module.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("DEMO-2");

int
main(int argc, char* argv[])
{
  // Enable command-line argument parsing
  CommandLine cmd(__FILE__);
  cmd.Parse(argc, argv);

  // Set time resolution and enable some logging
  Time::SetResolution(Time::NS);
  LogComponentEnable("UdpEchoClientApplication", LOG_LEVEL_INFO);
  LogComponentEnable("UdpEchoServerApplication", LOG_LEVEL_INFO);

  // Create 3 nodes: n0, n1, n2
  NodeContainer allNodes;
  allNodes.Create(3);

  // 1) First link: n0 <--> n1
  NodeContainer nodes1;
  nodes1.Add(allNodes.Get(0)); // n0
  nodes1.Add(allNodes.Get(1)); // n1

  // 2) Second link: n1 <--> n2
  NodeContainer nodes2;
  nodes2.Add(allNodes.Get(1)); // n1
  nodes2.Add(allNodes.Get(2)); // n2

  // 3) Third link: n0 <--> n2
  NodeContainer nodes3;
  nodes3.Add(allNodes.Get(0)); // n0
  nodes3.Add(allNodes.Get(2)); // n2

  // Define PointToPoint helpers for each link
  PointToPointHelper p2p1, p2p2, p2p3;

  // p2p1: n0 <-> n1
  p2p1.SetDeviceAttribute("DataRate", StringValue("5Mbps"));
  p2p1.SetChannelAttribute("Delay", StringValue("2ms"));

  // p2p2: n1 <-> n2
  p2p2.SetDeviceAttribute("DataRate", StringValue("3Mbps"));
  p2p2.SetChannelAttribute("Delay", StringValue("5ms"));

  // p2p3: n0 <-> n2
  p2p3.SetDeviceAttribute("DataRate", StringValue("2Mbps"));
  p2p3.SetChannelAttribute("Delay", StringValue("10ms"));

  // Install net devices for each link
  NetDeviceContainer devices1, devices2, devices3;
  devices1 = p2p1.Install(nodes1);
  devices2 = p2p2.Install(nodes2);
  devices3 = p2p3.Install(nodes3);

  // Enable ASCII and pcap tracing for each link
  AsciiTraceHelper ascii;
  p2p1.EnableAsciiAll(ascii.CreateFileStream("demo2-link1.tr"));
  p2p2.EnableAsciiAll(ascii.CreateFileStream("demo2-link2.tr"));
  p2p3.EnableAsciiAll(ascii.CreateFileStream("demo2-link3.tr"));

  p2p1.EnablePcapAll("demo2-link1");
  p2p2.EnablePcapAll("demo2-link2");
  p2p3.EnablePcapAll("demo2-link3");

  // Install the Internet stack on all nodes
  InternetStackHelper stack;
  stack.Install(allNodes);

  // Assign IPv4 addresses:
  //   p2p1 -> 10.1.1.0
  //   p2p2 -> 10.1.2.0
  //   p2p3 -> 10.1.3.0
  Ipv4AddressHelper address1, address2, address3;

  address1.SetBase("10.1.1.0", "255.255.255.0");
  Ipv4InterfaceContainer interfaces1 = address1.Assign(devices1);

  address2.SetBase("10.1.2.0", "255.255.255.0");
  Ipv4InterfaceContainer interfaces2 = address2.Assign(devices2);

  address3.SetBase("10.1.3.0", "255.255.255.0");
  Ipv4InterfaceContainer interfaces3 = address3.Assign(devices3);

  // Populate routing tables so that packets can find all routes
  Ipv4GlobalRoutingHelper::PopulateRoutingTables();

  // Create a UDP Echo Server on node n2 (port 9)
  UdpEchoServerHelper echoServer(9);
  ApplicationContainer serverApps = echoServer.Install(allNodes.Get(2));
  serverApps.Start(Seconds(1.0));
  serverApps.Stop(Seconds(10.0));

  // Create a UDP Echo Client on node n0, targeting the n2 server
  // The server is reachable at both 10.1.2.2 (via n1) and 10.1.3.2 (direct n0->n2 link).
  // We'll pick the direct link address (interfaces3.GetAddress(1)) for demonstration.
  UdpEchoClientHelper echoClient(interfaces3.GetAddress(1), 9);
  echoClient.SetAttribute("MaxPackets", UintegerValue(5));
  echoClient.SetAttribute("Interval", TimeValue(Seconds(1.0)));
  echoClient.SetAttribute("PacketSize", UintegerValue(512));

  ApplicationContainer clientApps = echoClient.Install(allNodes.Get(0)); // n0
  clientApps.Start(Seconds(2.0));
  clientApps.Stop(Seconds(10.0));

  // Run the simulation
  Simulator::Run();
  Simulator::Destroy();

  return 0;
}

```

**Script:**

```python
#
# SPDX-License-Identifier: GPL-2.0-only
#
# Converted from C++ to Python
#

try:
    from ns import ns
except ModuleNotFoundError:
    raise SystemExit(
        "Error: ns3 Python module not found;"
        " Python bindings may not be enabled"
        " or your PYTHONPATH might not be properly configured"
    )

import sys

# Enable logging for Echo applications
ns.LogComponentEnable("UdpEchoClientApplication", ns.LOG_LEVEL_INFO)
ns.LogComponentEnable("UdpEchoServerApplication", ns.LOG_LEVEL_INFO)

# Creating the nodes
allNodes = ns.NodeContainer()
allNodes.Create(3)

# Define n0 <-> n1 link
nodes1 = ns.NodeContainer()
nodes1.Add(allNodes.Get(0))
nodes1.Add(allNodes.Get(1))

# Define n1 <-> n2 link
nodes2 = ns.NodeContainer()
nodes2.Add(allNodes.Get(1))
nodes2.Add(allNodes.Get(2))

# Define n0 <-> n2 link
nodes3 = ns.NodeContainer()
nodes3.Add(allNodes.Get(0))
nodes3.Add(allNodes.Get(2))

# Configuring the point-to-point links
p2p1 = ns.PointToPointHelper()
p2p1.SetDeviceAttribute("DataRate", ns.StringValue("5Mbps"))
p2p1.SetChannelAttribute("Delay", ns.StringValue("2ms"))

p2p2 = ns.PointToPointHelper()
p2p2.SetDeviceAttribute("DataRate", ns.StringValue("3Mbps"))
p2p2.SetChannelAttribute("Delay", ns.StringValue("5ms"))

p2p3 = ns.PointToPointHelper()
p2p3.SetDeviceAttribute("DataRate", ns.StringValue("2Mbps"))
p2p3.SetChannelAttribute("Delay", ns.StringValue("10ms"))

# Installing the devices and links
devices1 = p2p1.Install(nodes1)
devices2 = p2p2.Install(nodes2)
devices3 = p2p3.Install(nodes3)

# Enable tracing
ascii = ns.AsciiTraceHelper()
p2p1.EnableAsciiAll(ascii.CreateFileStream("demo2-link1.tr"))
p2p2.EnableAsciiAll(ascii.CreateFileStream("demo2-link2.tr"))
p2p3.EnableAsciiAll(ascii.CreateFileStream("demo2-link3.tr"))

p2p1.EnablePcapAll("demo2-link1")
p2p2.EnablePcapAll("demo2-link2")
p2p3.EnablePcapAll("demo2-link3")

# Installing Internet stack on all nodes
stack = ns.InternetStackHelper()
stack.Install(allNodes)

# Assigning IP addresses
address1 = ns.Ipv4AddressHelper()
address1.SetBase(ns.Ipv4Address("10.1.1.0"), ns.Ipv4Mask("255.255.255.0"))
interfaces1 = address1.Assign(devices1)

address2 = ns.Ipv4AddressHelper()
address2.SetBase(ns.Ipv4Address("10.1.2.0"), ns.Ipv4Mask("255.255.255.0"))
interfaces2 = address2.Assign(devices2)

address3 = ns.Ipv4AddressHelper()
address3.SetBase(ns.Ipv4Address("10.1.3.0"), ns.Ipv4Mask("255.255.255.0"))
interfaces3 = address3.Assign(devices3)

# UdpEchoServer on node n2
echoServer = ns.UdpEchoServerHelper(9)
serverApps = echoServer.Install(allNodes.Get(2))
serverApps.Start(ns.Seconds(1.0))
serverApps.Stop(ns.Seconds(10.0))

# UdpEchoClient on node n0, targeting n2
echoClient = ns.UdpEchoClientHelper(interfaces3.GetAddress(1).ConvertTo(), 9)
echoClient.SetAttribute("MaxPackets", ns.UintegerValue(5))
echoClient.SetAttribute("Interval", ns.TimeValue(ns.Seconds(1.0)))
echoClient.SetAttribute("PacketSize", ns.UintegerValue(512))

clientApps = echoClient.Install(allNodes.Get(0))
clientApps.Start(ns.Seconds(2.0))
clientApps.Stop(ns.Seconds(10.0))

# Populating routing tables
ns.Ipv4GlobalRoutingHelper.PopulateRoutingTables()

# Run the simulation
ns.Simulator.Run()
ns.Simulator.Destroy()
```
