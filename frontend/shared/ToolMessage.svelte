<script lang="ts">
    import Tool from "./Tool.svelte";

    export let title: string;
    let expanded = false;
  
    function toggleExpanded() {
      expanded = !expanded;
    }
  </script>
  
<!-- svelte-ignore a11y-click-events-have-key-events -->
<!-- svelte-ignore a11y-no-static-element-interactions -->
<div class="box" on:click={toggleExpanded}>
    <div class="title">
        <div class="wrench-icon">
            🛠️
        </div>
        <span>{title}</span>
    </div>
    {#if expanded}
        <div class="content">
        <slot></slot>
        </div>
    {/if}
</div>
  
<style>
    .box {
        border-radius: 4px;
        cursor: pointer;
        max-width: max-content;
        background: var(--color-accent-soft)
    }

    .title {
        display: flex;
        align-items: center;
        padding: 8px;
    }

    .wrench-icon {
        width: 18px;
        height: 18px;
        margin-right: 8px;
    }

    .icon {
        width: 0; 
        height: 0; 
        border-left: 5px solid transparent;
        border-right: 5px solid transparent;
        border-top: 5px solid purple;
        margin-left: auto;
        transition: transform 0.3s;
    }

    .icon.expanded {
        transform: rotate(180deg);
    }

    .content {
        padding: 8px;
    }
</style>