// Azure Bicep — deploys Multi-Agent Day Planner to Container Apps
@description('Location for all resources')
param location string = resourceGroup().location

@description('Environment tag')
param environment string = 'prod'

@secure()
param openaiApiKey string

@secure()
param groqApiKey string

var appName = 'multi-agent-planner'
var tags = { environment: environment, project: appName }

resource acr 'Microsoft.ContainerRegistry/registries@2023-07-01' = {
  name: '${replace(appName, "-", "")}${environment}acr'
  location: location
  tags: tags
  sku: { name: 'Basic' }
  properties: { adminUserEnabled: true }
}

resource redis 'Microsoft.Cache/redis@2023-08-01' = {
  name: '${appName}-${environment}-redis'
  location: location
  tags: tags
  properties: {
    sku: { name: 'Basic', family: 'C', capacity: 0 }
    enableNonSslPort: false
    minimumTlsVersion: '1.2'
  }
}

resource logAnalytics 'Microsoft.OperationalInsights/workspaces@2023-09-01' = {
  name: '${appName}-${environment}-logs'
  location: location
  tags: tags
  properties: { sku: { name: 'PerGB2018' }, retentionInDays: 30 }
}

resource caEnv 'Microsoft.App/managedEnvironments@2023-11-02-preview' = {
  name: '${appName}-${environment}-env'
  location: location
  tags: tags
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: logAnalytics.properties.customerId
        sharedKey: logAnalytics.listKeys().primarySharedKey
      }
    }
  }
}

resource apiApp 'Microsoft.App/containerApps@2023-11-02-preview' = {
  name: '${appName}-${environment}-api'
  location: location
  tags: tags
  properties: {
    managedEnvironmentId: caEnv.id
    configuration: {
      ingress: { external: true, targetPort: 8000, transport: 'http' }
      secrets: [
        { name: 'openai-key', value: openaiApiKey }
        { name: 'groq-key',   value: groqApiKey }
        { name: 'redis-url',  value: 'rediss://:${redis.listKeys().primaryKey}@${redis.properties.hostName}:6380/0' }
      ]
    }
    template: {
      containers: [{
        name: 'api'
        image: '${acr.properties.loginServer}/${appName}:latest'
        resources: { cpu: json('0.5'), memory: '1Gi' }
        env: [
          { name: 'OPENAI_API_KEY', secretRef: 'openai-key' }
          { name: 'GROQ_API_KEY',   secretRef: 'groq-key' }
          { name: 'REDIS_URL',      secretRef: 'redis-url' }
          { name: 'ENV',            value: environment }
        ]
      }]
      scale: { minReplicas: 1, maxReplicas: 5 }
    }
  }
}

resource streamlitApp 'Microsoft.App/containerApps@2023-11-02-preview' = {
  name: '${appName}-${environment}-ui'
  location: location
  tags: tags
  properties: {
    managedEnvironmentId: caEnv.id
    configuration: {
      ingress: { external: true, targetPort: 8501, transport: 'http' }
    }
    template: {
      containers: [{
        name: 'streamlit'
        image: '${acr.properties.loginServer}/${appName}-ui:latest'
        resources: { cpu: json('0.25'), memory: '0.5Gi' }
        env: [{ name: 'API_BASE', value: 'https://${apiApp.properties.configuration.ingress.fqdn}' }]
      }]
      scale: { minReplicas: 1, maxReplicas: 3 }
    }
  }
}

output apiUrl       string = 'https://${apiApp.properties.configuration.ingress.fqdn}'
output uiUrl        string = 'https://${streamlitApp.properties.configuration.ingress.fqdn}'
output acrServer    string = acr.properties.loginServer
