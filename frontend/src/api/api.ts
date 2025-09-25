import { chatHistorySampleData } from '../constants/chatHistory'
import {
  ChatMessage,
  Conversation,
  ConversationRequest,
  CosmosDBHealth,
  CosmosDBStatus,
  UserInfo
} from './models'

// 👉 Si la API vive en el mismo dominio, deja cadena vacía.
//    Si está en otro dominio/subdominio, pon la URL completa, p.ej. 'https://api.tu-dominio.com'
const API_BASE = ''

// Helper para garantizar envío de cookies (EasyAuth) y facilitar CORS cuando API_BASE sea externo.
function apiFetch(input: string, init: RequestInit = {}) {
  return fetch(`${API_BASE}${input}`, {
    credentials: 'include',         // 🔑 necesario para que viaje la cookie de EasyAuth
    ...init
  })
}

/* ========== Conversación ========== */

export async function conversationApi(options: ConversationRequest, abortSignal: AbortSignal): Promise<Response> {
  return apiFetch('/conversation', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ messages: options.messages }),
    signal: abortSignal
  })
}

/* ========== Autenticación (EasyAuth) ========== */

export async function getUserInfo(): Promise<UserInfo[]> {
  try {
    const res = await apiFetch('/.auth/me')

    // Si el App Service está en "Redirect to login", no suele llegar aquí con 401,
    // pero si está en "HTTP 401", lo gestionamos manualmente:
    if (res.status === 401) {
      console.log('No autenticado. Redirigiendo a login...')
      const returnUrl = encodeURIComponent(window.location.href)
      // Cambia 'aad' por 'google' / 'facebook' si usas otro proveedor.
      window.location.href = `/.auth/login/aad?post_login_redirect_uri=${returnUrl}`
      return []
    }

    if (!res.ok) {
      console.log('No identity provider found. Access to chat will be blocked.')
      return []
    }

    const payload = (await res.json()) as UserInfo[]
    return Array.isArray(payload) ? payload : []
  } catch (e) {
    console.error('Error resolving /.auth/me: ', e)
    return []
  }
}

/* ========== Historial (mock inicial) ========== */

// export const fetchChatHistoryInit = async (): Promise<Conversation[] | null> => {
export const fetchChatHistoryInit = (): Conversation[] | null => {
  return chatHistorySampleData
}

/* ========== Historial (API) ========== */

export const historyList = async (offset = 0): Promise<Conversation[] | null> => {
  try {
    const res = await apiFetch(`/history/list?offset=${offset}`, { method: 'GET' })
    const payload = await res.json()
    if (!Array.isArray(payload)) {
      console.error('There was an issue fetching your data.')
      return null
    }

    const conversations: Conversation[] = await Promise.all(
      payload.map(async (conv: any) => {
        const convMessages: ChatMessage[] = await historyRead(conv.id).catch(err => {
          console.error('error fetching messages: ', err)
          return []
        })
        const conversation: Conversation = {
          id: conv.id,
          title: conv.title,
          date: conv.createdAt,
          messages: convMessages
        }
        return conversation
      })
    )

    return conversations
  } catch (_err) {
    console.error('There was an issue fetching your data.')
    return null
  }
}

export const historyRead = async (convId: string): Promise<ChatMessage[]> => {
  try {
    const res = await apiFetch('/history/read', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ conversation_id: convId })
    })

    if (!res) return []
    const payload = await res.json()
    const messages: ChatMessage[] = []

    if (payload?.messages) {
      payload.messages.forEach((msg: any) => {
        const message: ChatMessage = {
          id: msg.id,
          role: msg.role,
          date: msg.createdAt,
          content: msg.content,
          feedback: msg.feedback ?? undefined
        }
        messages.push(message)
      })
    }
    return messages
  } catch (_err) {
    console.error('There was an issue fetching your data.')
    return []
  }
}

export const historyGenerate = async (
  options: ConversationRequest,
  abortSignal: AbortSignal,
  convId?: string
): Promise<Response> => {
  try {
    const body = convId
      ? JSON.stringify({ conversation_id: convId, messages: options.messages })
      : JSON.stringify({ messages: options.messages })

    const res = await apiFetch('/history/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body,
      signal: abortSignal
    })
    return res
  } catch (_err) {
    console.error('There was an issue fetching your data.')
    return new Response()
  }
}

export const historyUpdate = async (messages: ChatMessage[], convId: string): Promise<Response> => {
  try {
    const res = await apiFetch('/history/update', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ conversation_id: convId, messages })
    })
    return res
  } catch (_err) {
    console.error('There was an issue fetching your data.')
    const errRes: Response = { ...new Response(), ok: false, status: 500 }
    return errRes
  }
}

export const historyDelete = async (convId: string): Promise<Response> => {
  try {
    const res = await apiFetch('/history/delete', {
      method: 'DELETE',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ conversation_id: convId })
    })
    return res
  } catch (_err) {
    console.error('There was an issue fetching your data.')
    const errRes: Response = { ...new Response(), ok: false, status: 500 }
    return errRes
  }
}

export const historyDeleteAll = async (): Promise<Response> => {
  try {
    const res = await apiFetch('/history/delete_all', {
      method: 'DELETE',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({})
    })
    return res
  } catch (_err) {
    console.error('There was an issue fetching your data.')
    const errRes: Response = { ...new Response(), ok: false, status: 500 }
    return errRes
  }
}

export const historyClear = async (convId: string): Promise<Response> => {
  try {
    const res = await apiFetch('/history/clear', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ conversation_id: convId })
    })
    return res
  } catch (_err) {
    console.error('There was an issue fetching your data.')
    const errRes: Response = { ...new Response(), ok: false, status: 500 }
    return errRes
  }
}

export const historyRename = async (convId: string, title: string): Promise<Response> => {
  try {
    const res = await apiFetch('/history/rename', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ conversation_id: convId, title })
    })
    return res
  } catch (_err) {
    console.error('There was an issue fetching your data.')
    const errRes: Response = { ...new Response(), ok: false, status: 500 }
    return errRes
  }
}

/* ========== Health de Cosmos DB ========== */

export const historyEnsure = async (): Promise<CosmosDBHealth> => {
  try {
    const res = await apiFetch('/history/ensure', { method: 'GET' })
    const respJson = await res.json()

    let formattedResponse
    if (respJson.message) {
      formattedResponse = CosmosDBStatus.Working
    } else {
      if (res.status === 500) {
        formattedResponse = CosmosDBStatus.NotWorking
      } else if (res.status === 401) {
        formattedResponse = CosmosDBStatus.InvalidCredentials
      } else if (res.status === 422) {
        formattedResponse = respJson.error
      } else {
        formattedResponse = CosmosDBStatus.NotConfigured
      }
    }

    if (!res.ok) {
      return { cosmosDB: false, status: formattedResponse }
    } else {
      return { cosmosDB: true, status: formattedResponse }
    }
  } catch (err) {
    console.error('There was an issue fetching your data.')
    return { cosmosDB: false, status: err as any }
  }
}

/* ========== Settings del frontend ========== */

export const frontendSettings = async (): Promise<Response | null> => {
  try {
    const res = await apiFetch('/frontend_settings', { method: 'GET' })
    return await res.json()
  } catch (_err) {
    console.error('There was an issue fetching your data.')
    return null
  }
}

/* ========== Feedback de mensajes ========== */

export const historyMessageFeedback = async (messageId: string, feedback: string): Promise<Response> => {
  try {
    const res = await apiFetch('/history/message_feedback', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message_id: messageId, message_feedback: feedback })
    })
    return res
  } catch (_err) {
    console.error('There was an issue logging feedback.')
    const errRes: Response = { ...new Response(), ok: false, status: 500 }
    return errRes
  }
}
