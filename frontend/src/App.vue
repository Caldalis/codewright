<script setup lang="ts">
import { onMounted } from "vue";

import Composer from "./components/Composer.vue";
import MessageList from "./components/MessageList.vue";
import PlanPanel from "./components/PlanPanel.vue";
import Sidebar from "./components/Sidebar.vue";
import TopBar from "./components/TopBar.vue";
import { useConversation } from "./stores/conversation";

const conversation = useConversation();

// 页面加载即开一个新会话(空会话不污染侧栏列表:无标题的不显示)
onMounted(() => void conversation.openSession(null));
</script>

<template>
  <div class="flex h-full">
    <Sidebar />
    <div class="flex min-w-0 flex-1 flex-col">
      <TopBar />
      <MessageList />
      <Composer />
    </div>
    <PlanPanel />
  </div>
</template>
