# Cloud technologies
This is a practice sheet for PROG2005 exam (cloud technologies)

### Principles of Cloud Computing
Cloud computing comes in different models based on what is being delivered and who manages it.

**--Types of cloud Computing--**
**IaaS**: (Infrastructure as a service)
Provides virtualized computing resources over the internet. Exampeles (OS's)

**PaaS**: (Platform as a service)
Provides a platform allowing users to develop, run, and manage applications. Exampels (Render, Firebase)

**SaaS**: (Software as a service)
Delivers software over the internet, no local install needed. Examples (Google Docs, Gmail)

**--Key considerations--**

When adopting cloud computing, organizations consider:
**Cost efficiency**: Pay as you go model reduces upfront cost

**Scalability**: Automatically scale resources based on demand

**Security**: Sensetive data must be protected

**Performance**: Geographically distributed servers can reduce latency
**Availability and reliability**: Redundancy  and failover are essential

**Vendor lock-in**: Migrating away from a provider can be difficult.

**--Motivation for Cloud Adoption--**
**Agility and speed**: Rapid development and deployment

**Global reach**: Access services from anywhere.

**Focus on core business**: Offload infrastructure  maintenance

**Environmental sustainability**: Shared infrastructure can reduce energy use.

### REST Principles and Standardisation

**--What is REST--**
**REST**: (Representational State Teransfer) is an architectural style for designing networked applications. It uses standard HTTP methods and is stateless.

**--Key Principles of REST--**
**Client-server**: the client and server are seperate; the client requests, the server responds. Promotes separation of concerns.

**Stateless**: Each request form a client must contain all the information needed to process it *no session state stored on the server)

**Cachable**: REsponses must define whether they are cachable to improve performance

**Uniform Interface**: A standardized way to interact with resources (e.g., consistent URIs, HTTP verbs)

**Layerd System**: Architecture composed of hierarchichal layers (e.g., load balancers, proxies)

**Code on demand**: Servers can temporarily extend client functionality bvy sending executable code (e.g., javascript)

**--Uniform Interface Constaraints--**
The **uniform interface** is the heart of REST. It simplifies and decouples the architecture.

**Resource identification**: Re sources are identifies in request (e.g., /users/123)

**Resource manipulation via representations**: Use JSON, XML, etc. to represent and manipulate resources.

**Self-discriptive messages**:  Each message includes enough information to describe how to process it

**Hypermedia as the engine of application state (HATEOAS)**: Clinets use hpyerlinks in repsonses to navigate the application.

**--Standard HTTP Methods in REST--**
* GET
* POST
* PUT
* PATCH
* DELETE





















